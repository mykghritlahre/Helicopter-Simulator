import argparse

import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve

from utilities import (
    ISA,
    RotorBlade,
    TurboTechEngine,
    Wing,
    color,
    g,
    parse_toml_params,
)


class Helicopter:
    """
    Helicopter class to simulate the helicopter dynamics

    Args:
    -----
    params: dict
        Dictionary containing the parameters of the helicopter
    """

    def __init__(
        self,
        design_params: dict,
    ) -> None:
        main_rotor_params: dict = design_params["rotors"]["main"]
        tail_rotor_params: dict = design_params["rotors"]["tail"]
        horizontal_stabilizer_params: dict = design_params["stabilizers"]["horizontal"]
        vertical_stabilizer_params: dict = design_params["stabilizers"]["vertical"]
        self.main_rotor = RotorBlade(main_rotor_params)
        self.tail_rotor = RotorBlade(tail_rotor_params)
        self.horizontal_stabilizer = Wing(horizontal_stabilizer_params)
        self.vertical_stabilizer = Wing(vertical_stabilizer_params)
        self.engine = TurboTechEngine()

        ### Constants ###
        self.empty_weight: float = design_params["helicopter"]["empty_weight"]
        self.cg_to_main_rotor: npt.NDArray = np.array(
            design_params["rotors"]["cg_to_main_rotor"]
        )
        self.cg_to_tail_rotor: npt.NDArray = np.array(
            design_params["rotors"]["cg_to_tail_rotor"]
        )
        self.cg_to_horizontal_stabilizer: npt.NDArray = np.array(
            design_params["stabilizers"]["cg_to_horizontal_stabilizer"]
        )
        self.cg_to_vertical_stabilizer: npt.NDArray = np.array(
            design_params["stabilizers"]["cg_to_vertical_stabilizer"]
        )
        self.alpha_fuselage: float = float(
            np.deg2rad(design_params["helicopter"]["alpha_fuselage"])
        )
        self.flat_plate_area: float = design_params["helicopter"]["flat_plate_area"]
        self.fuel_density: float = design_params["helicopter"]["fuel_density"]

        # emperical parameters for trim curve fit
        self.K: float = design_params["rotors"]["K"]
        self.Cd0: float = design_params["rotors"]["Cd0"]
        self.four_point_six: float = design_params["rotors"]["four_point_six"]
        self.m = design_params["rotors"]["m"]
        self.c = design_params["rotors"]["c"]
        self.dt = 0.5

        ### Variables ###
        self.psi_o: float = 0
        self.mass: float = self.empty_weight
        self.atmosphere = ISA()

    def update_atmosphere(
        self, altitude: float
    ) -> tuple[float, float, float, float, float]:
        """
        Update the atmosphere parameters. This is to be called at every time step

        Args:
        altitude: float
            Altitude of the helicopter

        Returns:
        tuple[float, float, float, float, float]
            Temperature, Pressure, Density, Speed of sound, Dynamic viscosity
        """
        return (
            self.atmosphere.temperature(altitude),
            self.atmosphere.pressure(altitude),
            self.atmosphere.density(altitude),
            self.atmosphere.speed_of_sound(altitude),
            self.atmosphere.dynamic_viscosity(altitude),
        )

    def update_mass(self, fuel: float, payload: float) -> None:
        """
        Update the mass of the helicopter externally -- used in mission.py

        Args:
        fuel: float
            Fuel mass
        payload: float
            Payload mass

        Returns:
        None
        """
        self.mass = self.empty_weight + fuel + payload

    def update_helicopter(
        self,
        altitude: float,
        collective_pitch: float,
        lateral_cyclic: float,
        longitudinal_cyclic: float,
        tail_rotor_collective: float,
        forward_velocity: float = 1,
        climb_velocity: float = 0,
    ) -> None:
        """
        Update the helicopter parameters at every time step.
        Calculates the net thrust, moments, power, fuel burn rate, etc.

        (Currently the fuel is not being updated)

        Args:
        altitude: float
            Altitude of the helicopter
        collective_pitch: float
            Collective pitch of the main rotor
        lateral_cyclic: float
            Lateral cyclic pitch of the main rotor
        longitudinal_cyclic: float
            Longitudinal cyclic pitch of the main rotor
        tail_rotor_collective: float
            Collective pitch of the tail rotor
        forward_velocity: float
            Forward velocity of the helicopter
        climb_velocity: float
            Climb velocity of the helicopter
        dt: float
            Time step

        Returns:
        None
        """
        V_infty = forward_velocity * np.array(
            [-np.cos(self.alpha_fuselage), 0, -np.sin(self.alpha_fuselage)]
        ) + climb_velocity * np.array(
            [np.sin(self.alpha_fuselage), 0, -np.cos(self.alpha_fuselage)]
        )
        self.atmosphere_params = self.update_atmosphere(altitude)
        self.main_rotor.get_atmosphere(self.atmosphere_params)
        self.main_rotor.forward_flight_update_params(
            collective_pitch,
            lateral_cyclic,
            longitudinal_cyclic,
            self.psi_o,
            V_infty,
        )
        self.main_rotor.forward_flight_params()
        self.tail_rotor.get_atmosphere(self.atmosphere_params)
        self.tail_rotor.bemt_update_params(tail_rotor_collective, V_infty)
        self.main_rotor_thrust = self.main_rotor.thrust
        self.tail_rotor_thrust = self.tail_rotor.bemt_thrust()
        self.main_rotor_torque, self.main_rotor_power = (
            self.main_rotor.torque,
            self.main_rotor.power,
        )
        self.tail_rotor.bemt_torque_and_power()
        self.tail_rotor_torque, self.tail_rotor_power = (
            self.tail_rotor.torque,
            self.tail_rotor.power,
        )
        self.total_power = self.main_rotor_power + self.tail_rotor_power
        self.shaft_power_available = (
            self.engine.power_delivered(
                altitude, self.atmosphere_params[0], self.atmosphere_params[0]
            )
            * 0.9  # 90% efficiency
            * 1000  # W to kW
        )
        self.fuel_burn_rate = (
            (self.total_power / 1000)
            * self.engine.sfc(
                altitude, self.atmosphere_params[0], self.atmosphere_params[0]
            )
            / 60
        )  # kg/min
        # self.fuel = self.fuel - (dt / 60) * self.fuel_burn_rate
        # self.mass = self.empty_weight + self.payload + self.fuel

        ## Stabilizer

        median_vel = np.median(self.main_rotor.V, axis=0)

        alpha_req = np.arctan2(
            self.cg_to_main_rotor[2] - self.cg_to_horizontal_stabilizer[2],
            self.cg_to_main_rotor[0]
            - self.cg_to_horizontal_stabilizer[0]
            - self.main_rotor.r
            + self.main_rotor.radius,
        )

        alpha_vel = np.arctan2(median_vel[:, 2], median_vel[:, 0])

        stab_v_infty_elements = np.where((np.abs(alpha_req) - np.abs(alpha_vel)) > 0.01)

        if len(stab_v_infty_elements[0]) == 0:
            stab_median_v_infty = V_infty
        else:
            stab_median_v_infty = np.median(median_vel(stab_v_infty_elements), axis=0)

        self.horizontal_stabilizer.get_atmosphere(self.atmosphere_params)
        self.vertical_stabilizer.get_atmosphere(self.atmosphere_params)
        self.horizontal_stabilizer.wing_params(stab_median_v_infty)
        self.vertical_stabilizer.wing_params(stab_median_v_infty)

    def cal_median_forward_flight_params(
        self,
        inputs: list[float],
        altitude: float,
        forward_velocity: float,
        climb_velocity: float,
        N_ITERS: int = 10,
        verbose: bool = False,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Calculate the median of the forces and moments acting on the helicopter
        by calculating forces, moments etc N_ITERS times and taking the median.
        Azimuthal angle is updated by 360/N_ITERS at each iteration to ensure full disk coverage.

        Args:
        inputs: list[float]
            List of inputs to the helicopter: [main rotor collective, lateral cyclic, longitudinal cyclic, tail rotor collective]
        altitude: float
            Altitude of the helicopter
        forward_velocity: float
            Forward velocity of the helicopter
        climb_velocity: float
            Climb velocity of the helicopter
        N_ITERS: int
            Number of iterations to calculate the median
        verbose: bool
            Print the forces and moments

        Returns:
        tuple[npt.NDArray, npt.NDArray]
            Net force and net moment acting on the helicopter
        """
        (
            median_force,
            median_moment,
            median_main_thrust,
            median_tail_thrust,
            median_tpp,
            median_beta,
            median_alpha_tpp,
            max_alpha,
            median_power,
            median_mr_power,
            median_tr_power,
        ) = ([], [], [], [], [], [], [], [], [], [], [])

        for _ in range(N_ITERS):
            (
                collective_pitch,
                lateral_cyclic,
                longitudinal_cyclic,
                tail_rotor_collective,
            ) = inputs

            self.update_helicopter(
                altitude,
                collective_pitch,
                lateral_cyclic,
                longitudinal_cyclic,
                tail_rotor_collective,
                forward_velocity,
                climb_velocity,
            )

            fuselage_drag = (
                0.5
                * self.atmosphere_params[2]
                * (forward_velocity**2 + climb_velocity**2)
                * self.flat_plate_area
            )

            horizontal_stabilizer_net_force = (
                self.horizontal_stabilizer.lift + self.horizontal_stabilizer.drag
            )

            vertical_stabilizer_net_force = (
                self.vertical_stabilizer.lift + self.vertical_stabilizer.drag
            )

            net_force = (
                self.main_rotor_thrust
                + horizontal_stabilizer_net_force
                + vertical_stabilizer_net_force
                + np.array([0, -self.tail_rotor_thrust, 0])
                + self.mass
                * g
                * np.array(
                    [np.sin(self.alpha_fuselage), 0, -np.cos(self.alpha_fuselage)]
                )
                - fuselage_drag
                * np.array(
                    [np.cos(self.alpha_fuselage), 0, np.sin(self.alpha_fuselage)]
                )
            )

            # Moment in vector format m = r_i x F_i + \tau_main + \tau_tail
            net_moment = (
                np.cross(self.cg_to_main_rotor, self.main_rotor_thrust)
                + np.cross(
                    self.cg_to_tail_rotor, np.array([0, -self.tail_rotor_thrust, 0])
                )
                + self.main_rotor_torque
                + np.array([0, self.tail_rotor_torque, 0])
                + np.cross(
                    self.cg_to_horizontal_stabilizer, horizontal_stabilizer_net_force
                )
                + np.cross(
                    self.cg_to_vertical_stabilizer, vertical_stabilizer_net_force
                )
            )

            median_force.append(net_force)
            median_moment.append(net_moment)
            median_main_thrust.append(self.main_rotor_thrust)
            median_tail_thrust.append(self.tail_rotor_thrust)
            median_tpp.append(self.main_rotor.tpp)
            median_beta.append(self.main_rotor.beta)
            max_alpha.append(np.max(self.main_rotor.linear_twist - self.main_rotor.phi))
            median_power.append(self.total_power)
            (
                median_alpha_tpp.append(self.main_rotor.alpha_tpp)
                if hasattr(self.main_rotor, "alpha_tpp")
                else median_alpha_tpp.append(np.nan)
            )
            median_mr_power.append(self.main_rotor_power)
            median_tr_power.append(self.tail_rotor_power)

            self.psi_o += 360 / N_ITERS

        net_force = np.median(median_force, axis=0)
        net_moment = np.median(median_moment, axis=0)
        main_thrust = np.median(median_main_thrust, axis=0)
        tail_thrust = np.median(median_tail_thrust, axis=0)
        tpp = np.median(median_tpp, axis=0)
        beta_o = np.median(median_beta)
        alpha_tpp = np.median(median_alpha_tpp, axis=0)
        alpha = np.mean(max_alpha)
        power = np.median(median_power, axis=0)
        C_T = np.linalg.norm(main_thrust) / (
            self.main_rotor.density
            * self.main_rotor.area
            * (self.main_rotor.radius * self.main_rotor.omega) ** 2
        )
        lambda_i = np.linalg.norm(
            (self.main_rotor.V - self.main_rotor.V_infty)
            / (self.main_rotor.radius * self.main_rotor.omega)
        )
        mu = (
            np.linalg.norm(self.main_rotor.V_infty)
            * np.cos(self.main_rotor.alpha_tpp)
            / (self.main_rotor.radius * self.main_rotor.omega)
        )
        C_p_parasitic = self.flat_plate_area * mu**3 / 2 / self.main_rotor.area
        main_rotor_power = np.median(median_mr_power, axis=0)
        tail_rotor_power = np.median(median_tr_power, axis=0)
        ## Empirical definitions

        thrust_ = np.sqrt(fuselage_drag**2 + (self.mass * g) ** 2)
        C_T_ = thrust_ / (
            self.main_rotor.density
            * self.main_rotor.area
            * (self.main_rotor.radius * self.main_rotor.omega) ** 2
        )
        alpha_tpp_ = np.arctan2(fuselage_drag, self.mass * g)
        mu_ = (
            np.linalg.norm(self.main_rotor.V_infty)
            * np.cos(alpha_tpp_)
            / (self.main_rotor.radius * self.main_rotor.omega)
        )

        def __lambda_i(x, mu, alpha_tpp, C_T):
            return (
                4 * x**4
                + 8 * x**3 * mu * np.tan(alpha_tpp)
                + 4 * x**2 * mu**2 * (1 + np.tan(alpha_tpp) ** 2)
                - C_T**2
            )

        lambda_i_ = fsolve(__lambda_i, 0.1, args=(mu_, alpha_tpp_, C_T_))[0]
        C_p_parasitic_ = self.flat_plate_area * mu_**3 / 2 / self.main_rotor.area

        solidity_main = (
            self.main_rotor.nblades
            * (self.main_rotor.root_chord + self.main_rotor.tip_chord)
            / (2 * np.pi * self.main_rotor.radius)
        )

        solidity_tail = (
            self.tail_rotor.nblades
            * (self.tail_rotor.root_chord + self.tail_rotor.tip_chord)
            / (2 * np.pi * self.tail_rotor.radius)
        )

        Cp_main = main_rotor_power / (
            self.main_rotor.density
            * self.main_rotor.area
            * self.main_rotor.radius
            * (self.main_rotor.omega * self.main_rotor.radius) ** 2
        )

        Cp_total = power / (
            self.main_rotor.density
            * self.main_rotor.area
            * self.main_rotor.radius
            * (self.main_rotor.omega * self.main_rotor.radius) ** 2
        )

        Cp_tail = tail_rotor_power / (
            self.tail_rotor.density
            * self.tail_rotor.area
            * self.tail_rotor.radius
            * (self.tail_rotor.omega * self.tail_rotor.radius) ** 2
        )


        if verbose:
            print("\nMain Rotor Collective:          ", collective_pitch)
            print("Main Rotor Lateral Cyclic:      ", lateral_cyclic)
            print("Main Rotor Longitudinal Cyclic: ", longitudinal_cyclic)
            print("Tail Rotor Collective:          ", tail_rotor_collective)
            print("Forward Velocity (m/s):         ", forward_velocity)
            print("\n")
            print("Net force (Hub Frame):   ", net_force)
            print("Net moment (Hub Frame):  ", net_moment)
            print()
            print(
                "Net force (Body Frame):  ",
                np.array([net_force[0], -net_force[1], -net_force[2]]),
            )  # hub frame to body frame
            print(
                "Net moment (Body Frame): ",
                np.array([net_moment[0], -net_moment[1], -net_moment[2]]),
            )  # hub frame to body frame
            print(f"Main Rotor Thrust:        {main_thrust}")
            print(f"Tail Rotor Thrust:        {tail_thrust:.5f}")
            print(f"Fuselage Drag:            {fuselage_drag:.5f}")

            if alpha > self.main_rotor.STALL_ANGLE:
                print(
                    f"{color.RED}Mean of Max Alpha:        {np.rad2deg(alpha):.5f}{color.END}"
                )
            else:
                print(f"Mean of Max Alpha:        {np.rad2deg(alpha):.5f}")

            print(f"Alpha TPP:                {np.rad2deg(alpha_tpp):.5f}")
            print(f"beta_o:                   {np.rad2deg(beta_o):.5f}")
            print(
                f"beta_1c:                  {np.rad2deg(np.arctan2(tpp[0], tpp[2])):.5f}"
            )
            print(
                f"beta_1s:                  {np.rad2deg(np.arctan2(tpp[1], tpp[2])):.5f}"
            )

            if power > self.shaft_power_available:
                print(f"{color.RED}Power:                    {power:.5f}{color.END}")
            else:
                print(f"Power:                    {power:.5f}")
            print(f"Power Available:          {self.shaft_power_available:.5f}")

            print()
            print(f"True Values")
            print(f"Coefficient of Thrust C_T:  {C_T:.10f}")
            print(f"Lambda Induced:       {lambda_i}")
            print(f"Parasitic Power Coefficient C_p_parasitic: {C_p_parasitic:.10f}")
            print(f"Mu: {mu}")

            print()
            print(f"Empirical Values")
            print(f"Coefficient of Thrust C_T:  {C_T_:10f}")
            print(f"Lambda Induced:       {lambda_i_}")
            print(f"Parasitic Power Coefficient C_p_parasitic: {C_p_parasitic_:.10f}")
            print(f"Mu: {mu_}")

            print()
            print(f"Solidity:  {solidity_main:.10f}")
            print(
                "C_P main rotor",
                main_rotor_power
                / (
                    self.main_rotor.density
                    * self.main_rotor.area
                    * self.main_rotor.radius
                    * (self.main_rotor.omega * self.main_rotor.radius) ** 2
                ),
            )
            print(
                "C_P total",
                power
                / (
                    self.main_rotor.density
                    * self.main_rotor.area
                    * self.main_rotor.radius
                    * (self.main_rotor.omega * self.main_rotor.radius) ** 2
                ),
            )
            print("Median main rotor power", main_rotor_power)
            print("Median total power", power)

        return net_force, net_moment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helicopter design")
    parser.add_argument(
        "--params",
        type=str,
        default="data/params.toml",
        help="path/to/the/parameters/file.toml",
    )

    args = parser.parse_args()

    hel = Helicopter(
        parse_toml_params(args.params),
    )
    hel.mass += 50  # kg -- payload and fuel

    # values are in degrees
    main_rotor_collective = 1.89
    lateral_cyclic = 1.55
    longitudinal_cyclic = 2.62
    tail_rotor_collective = 1.76

    altitude = 2000  # m
    forward_velocity = 40 * 5 / 18  # change from default of 50 * 5/18 m/s
    climb_velocity = 0  # m/s

    hel.cal_median_forward_flight_params(
        [
            main_rotor_collective,
            lateral_cyclic,
            longitudinal_cyclic,
            tail_rotor_collective,
        ],
        altitude,
        forward_velocity,
        climb_velocity,
        verbose=True,
        N_ITERS=100,
    )
