import os
import argparse
import operator
import shutil

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.optimize import fsolve

from helicopter import Helicopter
from utilities import (
    PI,
    RotorBlade,
    color,
    display_text,
    emperical_function,
    g,
    parse_toml_params,
)

params = {
    "figure.figsize": [9, 6],
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "axes.titlepad": 15,
    "font.size": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": True if shutil.which("latex") else False,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
}
plt.rcParams.update(params)


class Mission:
    def __init__(
        self, design_params: dict, mission_params: dict, output_dir: str
    ) -> None:
        self.hel = Helicopter(design_params)
        self.mission_params = mission_params["mission"]
        self.plot = mission_params["plot"]
        self.output_dir = output_dir
        self.distance = 0  # km
        self.time = 0  # seconds
        self.altitude_list = []
        self.gross_weight_list = []
        self.fuel_list = []
        self.fuel_burn_rate_list = []
        self.total_power_list = []
        self.available_power_list = []
        self.speed_list = []
        self.climb_rate_list = []
        self.distance_covered_list = []
        self.time_list = []

    def __collective_pitch(
        self,
        collective_pitch: float,
        rotor: RotorBlade,
        V_infty: npt.NDArray,
        force: float,
    ) -> float:
        """
        Get the difference in the thrust produced by the rotor and the force.
        Objective function for fsolve to get collective pitch values

        Args:
        collective_pitch: float
            Collective pitch of the rotor
        rotor: RotorBlade
            Rotor blade object
        V_infty: npt.NDArray
            Freestream velocity

        Returns:
        float
            Difference in the thrust and the force
        """
        atmosphere_params = self.hel.update_atmosphere(self.altitude)
        rotor.get_atmosphere(atmosphere_params)
        rotor.bemt_update_params(collective_pitch, V_infty)
        thrust = rotor.bemt_thrust()
        return thrust - force

    def __get_cp_from_force(
        self, rotor: RotorBlade, V_infty: npt.NDArray, force: float
    ) -> float:
        """
        Get the collective pitch by comparing the thrust prouced by the rotor to
        the force it would need to balance. This force is mg for main rotor and
        the main rotor torque for the tail rotor.

        Args:
        force: float
            Force that the rotor needs to balance
        V_infty: npt.NDArray
            Freestream velocity
        rotor: RotorBlade
            Rotor blade object

        Returns:
        float
            Collective pitch of the main rotor
        """
        return fsolve(self.__collective_pitch, 6, args=(rotor, V_infty, force))[0]

    def __lambda_i(self, x, mu, alpha_tpp, C_T) -> float:
        """
        Objective function for fsolve to get lambda_i values

        Args:
        x: float
            lambda_i
        mu: float
            Advance ratio
        alpha_tpp: float
            Angle of tip path plane
        C_T: float
            Thrust coefficient

        Returns:
        float
            ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        """
        return (
            4 * x**4
            + 8 * x**3 * mu * np.tan(alpha_tpp)
            + 4 * x**2 * mu**2 * (1 + np.tan(alpha_tpp) ** 2)
            - C_T**2
        )

    def store_data_for_plots(
        self, power: float, fuel_burn_rate: float, shaft_power_available: float
    ) -> None:
        """
        Store data for plots. Keep appending to the lists for each variable

        Args:
        power: float
            Power required by the helicopter
        fuel_burn_rate: float
            Fuel burn rate
        shaft_power_available: float
            Available power

        Returns:
        None
        """
        self.altitude_list.append(self.altitude)
        self.gross_weight_list.append(self.hel.mass)
        self.fuel_list.append(self.fuel)
        self.fuel_burn_rate_list.append(fuel_burn_rate)
        self.total_power_list.append(power)
        self.available_power_list.append(shaft_power_available)
        self.speed_list.append(self.forward_velocity)
        self.climb_rate_list.append(self.climb_velocity)
        self.distance_covered_list.append(self.distance)
        self.time_list.append(self.time)

    def create_plots(self) -> None:
        """
        Create plots for the mission profile. This is called if power required is
        more than available power or fuel is exhausted or at the end of a successful
        mission.

        Args:
        None

        Returns:
        None
        """
        if self.plot:
            for list, label in zip(
                [
                    self.altitude_list,
                    self.gross_weight_list,
                    self.fuel_list,
                    self.fuel_burn_rate_list,
                    np.array(self.speed_list) * 18 / 5,
                    self.climb_rate_list,
                    np.array(self.distance_covered_list) / 1000,
                ],
                [
                    "Altitude [m]",
                    "Gross Weight [kg]",
                    "Fuel [kg]",
                    "Fuel Burn Rate [kg/min]",
                    "Speed [km/h]",
                    "Climb Rate [m/s]",
                    "Distance Covered [km]",
                ],
            ):
                fig, ax = plt.subplots(1, 1)
                ax.plot(np.array(self.time_list) / 60, list)
                ax.set_xlabel("Time [min]")
                ax.set_ylabel(label)
                fig.savefig(
                    f"{self.output_dir}/{label.split('[')[0][:-1].replace(' ', '_')}.png"
                )
                plt.close()

            fig, ax = plt.subplots(1, 1)
            ax.plot(
                np.array(self.time_list) / 60,
                np.array(self.total_power_list) / 1000,
                label="Required Power",
            )
            ax.plot(
                np.array(self.time_list) / 60,
                np.array(self.available_power_list) / 1000,
                label="Available Power",
            )
            ax.set_xlabel("Time [min]")
            ax.set_ylabel("Power [kW]")
            ax.legend()
            fig.savefig(f"{self.output_dir}/Power.png")
            plt.close()

    def update_fuel(self, power: float, atmosphere_params: tuple, dt: float) -> None:
        """
        Update the fuel and mass of the helicopter and check if the fuel is exhausted

        Args:
        power: float
            Power required by the helicopter
        atmosphere_params: tuple
            Parameters of the atmosphere
        dt: float
            Time step

        Returns:
            None
        """
        shaft_power_available = (
            self.hel.engine.power_delivered(
                self.altitude, atmosphere_params[0], atmosphere_params[0]
            )
            * 0.9  # 90% efficiency
            * 1000  # W to kW
        )
        if power > shaft_power_available:
            self.create_plots()
            raise ValueError(
                color.RED + "Power required is more than available power" + color.END
            )
        fuel_burn_rate = (
            (power / 1000)
            * self.hel.engine.sfc(
                self.altitude, atmosphere_params[0], atmosphere_params[0]
            )
            / 60
        )  # kg / min
        self.fuel -= fuel_burn_rate * dt / 60  # dt is in seconds
        self.hel.update_mass(self.fuel, self.payload)
        self.store_data_for_plots(power, fuel_burn_rate, shaft_power_available)
        if self.fuel < 0:
            self.create_plots()
            raise ValueError(color.RED + "Fuel exhausted" + color.END)

    def takeoff(self, takeoff_params: dict) -> None:
        """
        Simulate the takeoff segment

        Args:
        takeoff_params: dict
            Parameters for the takeoff segment

        Returns:
        None
        """
        self.altitude = takeoff_params["altitude"]
        self.fuel = takeoff_params["fuel"]
        self.payload = takeoff_params["payload"]
        self.climb_velocity = 0  # m/s
        self.forward_velocity = 0  # m/s
        self.hel.update_mass(self.fuel, self.payload)
        atmosphere_params = self.hel.update_atmosphere(self.altitude)
        shaft_power_available = (
            self.hel.engine.power_delivered(
                self.altitude, atmosphere_params[0], atmosphere_params[0]
            )
            * 0.9  # 90% efficiency
            * 1000  # W to kW
        )
        self.store_data_for_plots(0, 0, shaft_power_available)

    def vertical_climb_descent_and_hover(self, params: dict, segment: str) -> None:
        """
        Simulate vertical climb, descent and hover segments

        Args:
        params: dict
            Parameters for the segment
        segment: str
            Type of segment

        Returns:
        None
        """
        target_altitude = params["altitude"]

        # maintain consistency in altitude from one mission profile to the next
        if segment == "hover":
            assert (
                self.altitude == target_altitude
            ), "Altitude from previous segment should be same as hover altitude"

        self.climb_velocity = (
            params["climb_velocity"] if "climb_velocity" in params else 0
        )  # m/s
        self.forward_velocity = 0  # m/s
        duration = params["duration"] if "duration" in params else np.inf  # seconds

        # dynamic time step based on altitude to be covered and climb velocity for
        # climb and descent, and hover duration for hover
        dt = (
            duration / 20
            if duration < np.inf
            else np.abs((self.altitude - target_altitude) / self.climb_velocity) / 20
        )  # seconds
        time = 0  # seconds

        # define the operator (<= or >=) based on the segment type
        if segment == "vertical_climb":
            op = operator.le
        else:
            op = operator.ge

        # we use target altitude for convergence in case of climb and descent,
        # and set the duration to np.inf for these segments, but for hover we
        # use hover duration for convergence
        while (op(self.altitude, target_altitude)) and (time < duration):
            atmosphere_params = self.hel.update_atmosphere(self.altitude)
            self.hel.main_rotor.get_atmosphere(atmosphere_params)
            main_rotor_v_infty = np.array([0, 0, -self.climb_velocity])

            # trim the main rotor using thrust = mg
            main_rotor_cp = self.__get_cp_from_force(
                self.hel.main_rotor, main_rotor_v_infty, self.hel.mass * g
            )
            self.hel.main_rotor.bemt_update_params(main_rotor_cp, main_rotor_v_infty)
            self.hel.main_rotor.bemt_torque_and_power()

            self.hel.tail_rotor.get_atmosphere(atmosphere_params)
            tail_rotor_v_infty = np.array([0, 0, 0])

            # trim the tail rotor using main_rotor_torque/R = tail_rotor_force
            tail_rotor_cp = self.__get_cp_from_force(
                self.hel.tail_rotor,
                tail_rotor_v_infty,
                self.hel.main_rotor.torque / np.abs(self.hel.cg_to_tail_rotor[0]),
            )
            self.hel.tail_rotor.bemt_update_params(tail_rotor_cp, tail_rotor_v_infty)
            self.hel.tail_rotor.bemt_torque_and_power()

            self.altitude += self.climb_velocity * dt
            time += dt
            self.time += dt
            total_power = self.hel.main_rotor.power + self.hel.tail_rotor.power
            self.update_fuel(total_power, atmosphere_params, dt)

        # set the altitude to the target altitude to maintain consistency from
        # one mission profile to the next
        self.altitude = target_altitude

    def steady_climb_descent_and_level_flight(
        self, flight_params: dict, segment: str
    ) -> None:
        """
        Simulate steady climb, descent and level flight segments

        Args:
        flight_params: dict
            Parameters for the flight segment
        segment: str
            Type of segment

        Returns:
        None
        """
        set_altitude = flight_params[
            "altitude"
        ]  # this is target altitude in case of steady climb / descent
        if segment == "level_flight":
            assert (
                self.altitude == set_altitude
            ), "Altitude from previous segment should be same as level flight altitude"

        self.forward_velocity = flight_params["forward_velocity"]
        wind = flight_params["wind"] if "wind" in flight_params else 0
        self.climb_velocity = (
            flight_params["climb_velocity"] if "climb_velocity" in flight_params else 0
        )
        V_infty = np.array([-self.forward_velocity + wind, 0, -self.climb_velocity])

        target_distance = (
            flight_params["distance"] * 1000  # km to m
            if "distance" in flight_params
            else self.forward_velocity
            * np.abs((set_altitude - self.altitude) / self.climb_velocity)
        )

        # dynamic time step based on distance to be covered and forward velocity
        dt = target_distance / self.forward_velocity / 20  # seconds

        # we will use forward distance covered instead of altitude convergence
        # so that its compatible with level flight as well
        distance_covered = 0
        while distance_covered < target_distance:
            atmosphere_params = self.hel.update_atmosphere(self.altitude)
            self.hel.main_rotor.get_atmosphere(atmosphere_params)
            self.hel.tail_rotor.get_atmosphere(atmosphere_params)

            # calculate estimated power required using simplified equations for C_T,
            # alpha_tpp, mu, and lambda_i
            density = atmosphere_params[2]
            flat_plate_drag = (
                0.5
                * density
                * ((self.forward_velocity - wind) ** 2 + self.climb_velocity**2)
                * self.hel.flat_plate_area
            )
            C_T = np.sqrt((self.hel.mass * g) ** 2 + flat_plate_drag**2) / (
                density
                * self.hel.main_rotor.area
                * (self.hel.main_rotor.omega * self.hel.main_rotor.radius) ** 2
            )
            alpha_tpp = np.arctan(flat_plate_drag / (self.hel.mass * g))
            mu = (
                np.linalg.norm(V_infty)
                * np.cos(alpha_tpp)
                / (self.hel.main_rotor.omega * self.hel.main_rotor.radius)
            )
            lambda_i = fsolve(self.__lambda_i, 0.1, args=(mu, alpha_tpp, C_T))[0]

            # create the x vector for the emperical function fit
            x1 = C_T * lambda_i
            x2 = 0.5 * (self.hel.flat_plate_area / self.hel.main_rotor.area) * mu**3
            x3 = (
                self.hel.main_rotor.nblades
                * (self.hel.main_rotor.root_chord + self.hel.main_rotor.tip_chord)
                / (2 * PI * self.hel.main_rotor.radius)
            ) / 8
            x4 = mu**2
            x = np.array([x1, x2, x3, x4])

            C_P_main_rotor = emperical_function(
                x,
                self.hel.K,
                self.hel.Cd0,
                self.hel.four_point_six,
                "main",
            )
            main_rotor_power = C_P_main_rotor * (
                self.hel.main_rotor.density
                * self.hel.main_rotor.area
                * self.hel.main_rotor.radius
                * (self.hel.main_rotor.omega * self.hel.main_rotor.radius) ** 2
            )
            # power required by the tail rotor is taken as a function of the main
            # rotor power. Refer trim_curve.ipynb and the slides for details.
            C_P_tail_rotor = self.hel.m * C_P_main_rotor + self.hel.c
            tail_rotor_power = C_P_tail_rotor * (
                self.hel.tail_rotor.density
                * self.hel.tail_rotor.area
                * self.hel.tail_rotor.radius
                * (self.hel.tail_rotor.omega * self.hel.tail_rotor.radius) ** 2
            )
            total_power = main_rotor_power + tail_rotor_power

            self.altitude += self.climb_velocity * dt
            distance_covered += self.forward_velocity * dt
            self.distance += self.forward_velocity * dt
            self.time += dt
            self.update_fuel(total_power, atmosphere_params, dt)

        # set the altitude to the target altitude to maintain consistency from
        # one mission profile to the next
        self.altitude = set_altitude

    def change_payload(self, change_payload_params: dict) -> None:
        self.altitude = change_payload_params["altitude"]
        self.payload = change_payload_params["payload"]
        self.hel.update_mass(self.fuel, self.payload)
        atmosphere_params = self.hel.update_atmosphere(self.altitude)
        shaft_power_available = (
            self.hel.engine.power_delivered(
                self.altitude, atmosphere_params[0], atmosphere_params[0]
            )
            * 0.9  # 90% efficiency
            * 1000  # W to kW
        )
        self.store_data_for_plots(
            self.total_power_list[-1],
            self.fuel_burn_rate_list[-1],
            shaft_power_available,
        )

    def simulate(self):
        for segment in self.mission_params.keys():
            segment_params = self.mission_params[segment]
            segment_type = segment_params["type"]
            display_text(f"Starting {segment_type} segment", c="cyan")
            match segment_type:
                case "takeoff":
                    self.takeoff(segment_params)
                case "vertical_climb" | "vertical_descent" | "hover":
                    self.vertical_climb_descent_and_hover(segment_params, segment_type)
                case "level_flight" | "steady_climb" | "steady_descent":
                    self.steady_climb_descent_and_level_flight(
                        segment_params, segment_type
                    )
                case "change_payload":
                    self.change_payload(segment_params)
                case _:
                    raise ValueError(
                        f"{color.RED}Invalid segment {segment_type}{color.END}"
                    )
        print()
        display_text(f"Completed the Mission", c="green")
        self.create_plots()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Simulate the helicopter mission.""")
    parser.add_argument(
        "--hel_params",
        type=str,
        default="data/params.toml",
        help="path/to/the/helicopter/parameters/file.toml",
    )
    parser.add_argument(
        "--mission_params",
        type=str,
        default="data/mission_A.toml",
        help="path/to/the/mission/parameters/file.toml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./images/mission_A/",
        help="path/to/the/output/directory",
    )

    args = parser.parse_args()

    design_params = parse_toml_params(args.hel_params)
    mission_params = parse_toml_params(args.mission_params)

    os.makedirs(args.output_dir, exist_ok=True)
    mission = Mission(design_params, mission_params, args.output_dir)

    mission.simulate()
