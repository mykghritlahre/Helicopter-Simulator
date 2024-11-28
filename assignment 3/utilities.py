try:
    import tomllib
except ImportError:
    import toml as tomllib

import shutil

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import fsolve

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
np.set_printoptions(suppress=True, precision=5)

""" Define Constants """
PI = np.pi
g = 9.81  # m/s^2


def parse_toml_params(filename: str) -> dict:
    """
    Parse the toml file containing the parameters

    Parameters:
        -----------
    filename: str
        The path to the toml file

    Returns:
        --------
    dict
        The parameters in a dictionary format
    """
    with open(filename) as file:
        return tomllib.loads(file.read())


class color:
    """
    Class to define colors for text formatting
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def display_text(text: str, c: str = "yellow") -> None:
    """
    Display text in a colorful formatted manner

    Parameters:
        -----------
    text: str
        The text to be displayed

    Returns:
        --------
    None
    """
    colors = {
        "purple": color.PURPLE,
        "cyan": color.CYAN,
        "darkcyan": color.DARKCYAN,
        "blue": color.BLUE,
        "green": color.GREEN,
        "yellow": color.YELLOW,
        "red": color.RED,
    }
    print("#" + "-" * (10 + len(text)) + "#")
    print(
        "#"
        + ("-" * 5)
        + color.BOLD
        + colors[c]
        + color.UNDERLINE
        + str(text)
        + color.END
        + ("-" * 5)
        + "#"
    )
    print("#" + "-" * (10 + len(text)) + "#")


def emperical_function(
    x: npt.NDArray,
    kappa: npt.NDArray | float,
    Cd0: npt.NDArray | float,
    four_point_six: npt.NDArray | float,
    rotor_type: str,
) -> npt.NDArray | float:
    """
    Empirical function to fit the data

    Parameters
    ----------
    x : array_like
        The input data in the form [x1, x2, x3, x4]
    kappa : float
        The coefficient of the first term
    Cd0 : float
        The coefficient of the second term
    four_point_six : float
        The coefficient of the third term
    rotor_type : str
        The type of rotor (main or tail)

    Returns
    -------
    float
        The output of the function
    """
    if rotor_type == "main":
        x1, x2, x3, x4 = x
        return x1 * kappa + x2 + x3 * Cd0 * (1 + four_point_six * x4)
    elif rotor_type == "tail":
        x1, x2, x3 = x
        return x1 * kappa + x2 * Cd0 * (1 + four_point_six * x3)
    else:
        raise ValueError("Invalid rotor type")


class ISA:
    """
    International Standard Atmosphere
    =================================
    Class to define the International Standard Atmosphere

    Parameters:
        -----------
    ----
    path_to_csv: str (optional)
        Path to the ISA table in csv format

    Attributes:
    ------------
    isa_table: pd.DataFrame
        The ISA table in a pandas DataFrame format
    temperature: Callable[[float], float]
        Function to calculate the temperature at a given altitude
    pressure: Callable[[float], float]
        Function to calculate the pressure at a given altitude
    density: Callable[[float], float]
        Function to calculate the density at a given altitude
    speed_of_sound: Callable[[float], float]
        Function to calculate the speed of sound at a given altitude
    dynamic_viscosity: Callable[[float], float]
        Function to calculate the dynamic viscosity at a given altitude
    """

    def __init__(self, path_to_csv: str = "data/isa.csv") -> None:
        self.isa_table = pd.read_csv(path_to_csv)

    def temperature(self, altitude: float) -> float:
        return float(
            np.interp(
                altitude,
                self.isa_table["Height [m]"],
                self.isa_table["Temperature [K]"],
            )
        )

    def pressure(self, altitude: float) -> float:
        return float(
            np.interp(
                altitude, self.isa_table["Height [m]"], self.isa_table["Pressure [Pa]"]
            )
        )

    def density(self, altitude: float) -> float:
        return float(
            np.interp(
                altitude,
                self.isa_table["Height [m]"],
                self.isa_table["Density [kg/m^3]"],
            )
        )

    def speed_of_sound(self, altitude: float) -> float:
        return float(
            np.interp(
                altitude,
                self.isa_table["Height [m]"],
                self.isa_table["Speed of Sound [m/s]"],
            )
        )

    def dynamic_viscosity(self, altitude: float) -> float:
        return float(
            np.interp(
                altitude,
                self.isa_table["Height [m]"],
                self.isa_table["Dynamic Viscosity [kg/(m*s)]"],
            )
        )


class TurboTechEngine:
    """
    Turbo Tech Engine
    =================
    Class to define the turboprop engine

    Parameters:
    -----------
    path_to_csv: str (optioanl)
        Path to the csv file containing the turboprop data

    Attributes:
    -----------
    turboprop_table: `pd.DataFrame`
        The turboprop data in a pandas DataFrame format
    power_delivered: `Callable[[float, float, float], float]`
        Function to calculate the power delivered by the engine
    sfc: `Callable`[[float, float, float], float]`
        Function to calculate the specific fuel consumption
    fuel_flow: `Callable[[float, float, float], float]`
        Function to calculate the fuel flow
    """

    def __init__(self, path_to_csv: str = "data/turboprop_data.csv") -> None:
        self.turboprop_table = pd.read_csv(path_to_csv)

    def __interpolate(self, y: str) -> float:
        """
        Internal function to interpolate the turboprop data across altitudes and delta T

        Parameters:
        -----------
        y: str
            The column to interpolate

        Returns:
        --------
        float
            The interpolated value
        """
        altitudes = self.turboprop_table["Altitude"].unique()
        _y = []
        for alt in altitudes:
            temp = self.turboprop_table[self.turboprop_table["Altitude"] == alt]
            _y.append(
                np.interp(
                    self.delta_temp,
                    temp["Delta T from ISA"],
                    temp[y],
                )
            )
        return float(np.interp(self.altitude, altitudes, _y))

    def power_delivered(
        self, altitude: float, engine_temp: float, isa_temp: float
    ) -> float:
        """
        Function to calculate the power delivered by the engine

        Parameters:
        -----------
        altitude: float
            The altitude at which the engine is operating
        engine_temp: float
            The temperature of the engine
        isa_temp: float
            The temperature of the ISA at the given altitude

        Returns:
        --------
        float
            The power delivered by the engine in kW
        """
        self.delta_temp = engine_temp - isa_temp
        self.altitude = altitude
        return self.__interpolate("Prop Shaft Power Delivered")

    def sfc(self, altitude: float, engine_temp: float, isa_temp: float) -> float:
        """
        Function to calculate the specific fuel consumption

        Parameters:
        -----------
        altitude: float
            The altitude at which the engine is operating
        engine_temp: float
            The temperature of the engine
        isa_temp: float
            The temperature of the ISA at the given altitude

        Returns:
        --------
        float
            The specific fuel consumption in kg/kWh
        """
        self.delta_t = engine_temp - isa_temp
        self.altitude = altitude
        return self.__interpolate("SFC")

    def fuel_flow(self, altitude: float, engine_temp: float, isa_temp: float) -> float:
        """
        Function to calculate the fuel flow

        Parameters:
        -----------
        altitude: float
            The altitude at which the engine is operating
        engine_temp: float
            The temperature of the engine
        isa_temp: float
            The temperature of the ISA at the given altitude

        Returns:
        --------
        float
            The fuel flow in liters per hour
        """
        self.delta_temp = engine_temp - isa_temp
        self.altitude = altitude
        return self.__interpolate("Fuel flow")


class Wing:
    """
    Wing
    ====
    Class to define the stabilizing wings

    Parameters:
    -----------
    """

    def __init__(
        self,
        params: dict[str, float],
    ):
        self.type = params["type"]
        self.span = params["span"]
        self.root_chord = params["root_chord"]
        self.tip_chord = params["tip_chord"]
        self.aspect_ratio = 2 * self.span / (self.root_chord + self.tip_chord)
        self.area = self.span * (self.root_chord + self.tip_chord) / 2
        self.naca0012_cl = pd.read_csv("data/NACA0012_CL.csv")
        self.naca0012_cd = pd.read_csv("data/NACA0012_CD.csv")
        linear_region = self.naca0012_cl[self.naca0012_cl["alpha"] > 0][:15]
        self.Cl_alpha = (linear_region["cl"] / linear_region["alpha"]).median()

    def _Cl(self, alpha: float) -> float:
        """
        Function to calculate the lift coefficient of the wing

        Parameters:
        -----------
        alpha: npt.NDArray
            The angle of attack

        Returns:
        --------
        npt.NDArray
            The lift coefficient of the wing
        """
        return np.interp(
            alpha, self.naca0012_cl["alpha"], self.naca0012_cl["cl"]
        ) / np.sqrt(1 - self.mach_number**2)

    def _Cd(self, Cl: float) -> float:
        """
        Function to calculate the drag coefficient of the wing. This is
        calculated using the drag polar equation, so Cl is needed for it.

        Parameters:
        -----------
        Cl: float
            The lift coefficient of the wing

        Returns:
        --------
        float
            The drag coefficient of the wing
        """
        return np.interp(Cl, self.naca0012_cd["cl"], self.naca0012_cd["cd"]) / np.sqrt(
            1 - self.mach_number**2
        )

    def get_atmosphere(self, atmosphere_params: tuple) -> None:
        """
        Function to get the atmosphere parameters. The paramteres are to be passed
        in as tuple obatined from the ISA class.

        Parameters:
        -----------
        atmosphere_params: tuple
            Tuple containing the atmosphere parameters

        Returns:
        --------
        None
        """
        (
            self.temperature,
            self.pressure,
            self.density,
            self.speed_of_sound,
            self.dynamic_viscosity,
        ) = atmosphere_params

    def wing_params(
        self,
        V_infty: npt.NDArray,
    ) -> None:
        """
        Function to update the parameters of the wing

        Parameters:
        -----------
        V_infty: npt.NDArray
            The freestream velocity. For the wing, this
            can be the wake coming from the rotor.

        Returns:
        --------
        None
        """

        norm_V_infty = np.linalg.norm(V_infty)
        self.mach_number = norm_V_infty / self.speed_of_sound

        if self.type == "horizontal":
            alpha = np.arctan2(V_infty[2], V_infty[0])
            alpha = -(alpha + np.sign(V_infty[2] * V_infty[0]) * PI)
        elif self.type == "vertical":
            alpha = np.arctan2(V_infty[1], V_infty[0])
            alpha = -(alpha + np.sign(V_infty[1] * V_infty[0]) * PI)

        Cl = self._Cl(alpha)
        Cd = self._Cd(Cl)
        lift = 0.5 * self.density * (norm_V_infty) ** 2 * self.area * Cl
        drag = 0.5 * self.density * (norm_V_infty) ** 2 * self.area * Cd
        # get components perpendicular to wing
        self.lift = np.array([0, 0, lift * np.cos(alpha) - drag * np.sin(alpha)])
        self.drag = np.array([lift * np.sin(alpha) + drag * np.cos(alpha), 0, 0])


class RotorBlade:
    """
    Class to define any rotor blade

    Parameters:
    -----------
    params: dict[str, float]
        Dictionary containing the parameters of the rotor blade
    """

    def __init__(
        self,
        params: dict[str, float],
    ):
        ### User defined constants ###
        self.radius = params["radius"]
        self.root_cutout = params["root_cutout"]
        self.root_chord = params["root_chord"]
        self.tip_chord = params["tip_chord"]
        self.twist = np.deg2rad(params["twist"])
        self.nblades = int(params["nblades"])
        self.omega = params["omega"]
        self.area = PI * (self.radius**2)

        ### Assumed constants ###
        self.MIN_CLIMB_VELOCITY = 0.1
        self.NELEMENTS = 15
        self.blade_density = 2000  # kg/m^3
        self.psi_o = 0
        self.STALL_ANGLE = np.deg2rad(12)

        self.r = np.linspace(self.root_cutout, self.radius, self.NELEMENTS + 1)
        self.r = (self.r[1:] + self.r[:-1]) / 2
        self.dr = (self.radius - self.root_cutout) / self.NELEMENTS
        self.chord = (self.tip_chord - self.root_chord) * (
            self.r - self.root_cutout
        ) / (self.radius - self.root_cutout) + self.root_chord
        self.thickness = (
            0.15 * self.chord
        )  # 15% thickness: http://www.helistart.com/RotorBladeDesign.aspx
        self.element_mass = self.blade_density * self.chord * self.dr * self.thickness
        self.solidity = (
            self.nblades
            * (
                (self.tip_chord - self.root_chord)
                * (self.r - self.root_cutout)
                / (self.radius - self.root_chord)
                + self.root_chord
            )
            / (PI * self.radius)
        )

        ### Airfoil Data ###
        self.naca0012_cl = pd.read_csv("data/NACA0012_CL.csv")
        self.naca0012_cd = pd.read_csv("data/NACA0012_CD.csv")
        linear_region = self.naca0012_cl[self.naca0012_cl["alpha"] > 0][:15]
        self.Cl_alpha = (linear_region["cl"] / linear_region["alpha"]).median()

    def _twist_linear(self, total_pitch: float) -> npt.NDArray:
        """
        Function to calculate the linear twist of the rotor blade at each element

        Parameters:
        -----------
        total_pitch: float
            Total pitch (collective + cyclic) of the rotor blade

        Returns:
        --------
        npt.NDArray
            The linear twist of the rotor blade at each element
        """
        return self.twist * self.r + total_pitch

    def _Cl(self, linear_twist: npt.NDArray) -> npt.NDArray:
        """
        Function to calculate the lift coefficient of the rotor blade at each element

        Parameters:
        -----------
        linear_twist: npt.NDArray
            The linear twist of the rotor blade at each element

        Returns:
        --------
        npt.NDArray
            The lift coefficient of the rotor blade at each element
        """
        return np.interp(
            linear_twist - self.phi, self.naca0012_cl["alpha"], self.naca0012_cl["cl"]
        ) / np.sqrt(1 - self.mach_number**2)

    def _Cd(self, Cl: npt.NDArray | float) -> npt.NDArray | float:
        """
        Function to calculate the drag coefficient of the wing. This is
        calculated using the drag polar equation, so Cl is needed for it.

        Parameters:
        -----------
        Cl: float
            The lift coefficient of the wing

        Returns:
        --------
        float
            The drag coefficient of the wing
        """
        return np.interp(Cl, self.naca0012_cd["cl"], self.naca0012_cd["cd"]) / np.sqrt(
            1 - self.mach_number**2
        )

    def get_atmosphere(self, atmosphere_params: tuple) -> None:
        """
        Function to get the atmosphere parameters. The paramteres are to be passed
        in as tuple obatined from the ISA class.

        Parameters:
        -----------
        atmosphere_params: tuple
            Tuple containing the atmosphere parameters

        Returns:
        --------
        None
        """
        (
            self.temperature,
            self.pressure,
            self.density,
            self.speed_of_sound,
            self.dynamic_viscosity,
        ) = atmosphere_params

    def bemt_update_params(
        self,
        collective_pitch: float,
        V_infty: npt.NDArray,
    ) -> None:
        """
        Function to update the parameters of the rotor blade

        Parameters:
        -----------
        collective_pitch: float
            Collective pitch of the rotor blade in degrees
        V_infty: npt.NDArray
            The freestream velocity (climb velocity for BEMT)

        Returns:
        --------
        None
        """
        collective_pitch = np.deg2rad(collective_pitch)
        self.V_infty = V_infty
        self.linear_twist = self._twist_linear(collective_pitch)
        self.U_t = self.omega * self.r
        self.U_p = self._ptl() * self.omega * self.radius
        self.phi = np.arctan2(self.U_p, self.U_t)
        self.mach_number = np.sqrt(self.U_p**2 + self.U_t**2) / self.speed_of_sound
        if np.max(self.linear_twist - self.phi) > np.deg2rad(12):
            display_text("ROTOR TIPS ARE IN STALL")
        if np.max(self.mach_number) > 0.8:
            display_text("ROTOR TIPS ARE IN TRANSONIC REGIME")

    def _ptl(self) -> npt.NDArray:
        """
        Function implementing the Prandtl tip loss

        Parameters:
        -----------
        None

        Returns:
        --------
        npt.NDArray
            Inflow ratio, lambda = (V+v)/(omega*R)
        """
        if (not hasattr(self, "lam")) or (not hasattr(self, "lam_c")):
            lam_c = np.abs(self.V_infty[-1]) / (self.omega * self.radius)
            lam = (
                np.sqrt(
                    ((self.solidity * self.Cl_alpha / 16) - (lam_c / 2)) ** 2
                    + (
                        self.solidity
                        * self.Cl_alpha
                        * self.linear_twist
                        * self.r
                        / (8 * self.radius)
                    )
                )
                - (self.solidity * self.Cl_alpha / 16)
                + lam_c / 2
            )

        for _ in range(5):
            f = self.nblades * (1 - self.r / self.radius) / (2 * lam)
            F = (2 / PI) * np.arccos(np.exp(-f))
            if np.abs(self.V_infty[-1]) > self.MIN_CLIMB_VELOCITY:
                lam = (self.solidity * self.Cl_alpha / (16 * F)) * (
                    np.sqrt(
                        1
                        + (
                            32
                            * F
                            * self.linear_twist
                            * self.r
                            / (self.solidity * self.Cl_alpha * self.radius)
                        )
                    )
                    - 1
                )
            else:
                lam = (
                    np.sqrt(
                        (
                            (self.solidity * self.Cl_alpha / (16 * F) - (lam_c / 2))
                            ** 2
                            + (
                                self.solidity
                                * self.Cl_alpha
                                * self.linear_twist
                                * self.r
                                / (8 * F * self.radius)
                            )
                        )
                    )
                    - (self.solidity * self.Cl_alpha / (16 * F))
                    + lam_c / 2
                )
        return np.array(lam)

    def bemt_thrust(
        self,
    ) -> float:
        """Calculate the thrust using the Blade Element Momentum Theory

        Parameters:
        -----------
        None

        Returns:
        --------
        float
            The thrust
        """
        Cl = self._Cl(self.linear_twist)
        Cd = self._Cd(Cl)
        dT = (
            0.5
            * self.density
            * (self.U_t**2 + self.U_p**2)
            * self.chord
            * (Cl * np.cos(self.phi) - Cd * np.sin(self.phi))
        ) * self.dr
        self.thrust = self.nblades * np.sum(dT)
        self.Ct = (
            2
            * self.thrust
            / (self.density * self.area * (self.omega * self.radius) ** 2)
        )
        return self.thrust

    def bemt_torque_and_power(
        self,
    ) -> None:
        """Calculate the torque and power using the Blade Element Momentum Theory

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        Cl = self._Cl(self.linear_twist)
        Cd = self._Cd(Cl)
        dQ = (
            0.5
            * self.density
            * (self.U_t**2 + self.U_p**2)
            * self.chord
            * self.r
            * (Cl * np.sin(self.phi) + Cd * np.cos(self.phi))
        ) * self.dr
        self.torque = float(self.nblades * np.sum(dQ))
        self.power = self.torque * self.omega
        self.Cq = (
            2
            * self.torque
            / (self.density * self.area * self.radius * (self.omega * self.radius) ** 2)
        )
        self.CP = self.Cq

    def forward_flight_update_params(
        self,
        collective_pitch: float,
        lateral_cyclic: float,
        longitudinal_cyclic: float,
        psi_o: float,
        V_infty: npt.NDArray,
    ) -> None:
        """
        Update the parameters for forward flight

        Parameters:
        -----------
        collective_pitch: float
            Collective pitch of the rotor blade in degrees
        lateral_cyclic: float
            Lateral cyclic of the rotor blade in degrees (:math: `\theta_{1c}`)
        longitudinal_cyclic: float
            Longitudinal cyclic of the rotor blade in degrees (:math: `\theta_{1s}`)
        psi_o: float
            The azimuthal angle of the "main" blade upon which other blades are dependent
        forward_velocity: float
            Forward velocity of the helicopter

        Returns:
        --------
        None
        """
        theta_1c = np.deg2rad(lateral_cyclic)
        theta_1s = np.deg2rad(longitudinal_cyclic)
        collective_pitch = np.deg2rad(collective_pitch)
        self.psi_o = np.deg2rad(psi_o)
        self.psi = (
            np.deg2rad(np.arange(0, 360, 360 / self.nblades)) + self.psi_o
        )  # calculate psi for all blades at once
        total_pitch = (
            collective_pitch + theta_1c * np.cos(self.psi) + theta_1s * np.sin(self.psi)
        ).reshape(-1, 1)
        self.linear_twist = self._twist_linear(total_pitch)
        self.V_infty = V_infty

    def _blade_plane(self, beta) -> npt.NDArray:
        """Calculate the blade plane

        Parameters:
        -----------
        None

        Returns:
        --------
        npt.NDArray
            The blade plane
        """
        blade_plane = list()
        blade_plane.append(np.sin(beta) * np.cos(self.psi))
        blade_plane.append(np.sin(beta) * np.sin(self.psi))
        blade_plane.append(np.cos(beta))
        return np.array(blade_plane).T

    def _airfoil_plane(self, beta) -> npt.NDArray:
        """
        Calculate the airfoil plane

        Parameters:
        -----------
        None

        Returns:
        --------
        npt.NDArray
            The airfoil plane
        """
        airfoil_plane = list()
        airfoil_plane.append(np.cos(beta) * np.cos(self.psi))
        airfoil_plane.append(np.cos(beta) * np.sin(self.psi))
        airfoil_plane.append(-np.sin(beta))
        return np.array(airfoil_plane).T

    def _forward_flight_blade_forces(
        self, beta: npt.NDArray
    ) -> tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        """
        Internal function to calculate the blade forces in forward flight

        Parameters:
        -----------
        beta: npt.NDArray
            The coning angle

        Returns:
        --------
        tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
            The thrust, tangential force, normal vector, radius vector, cross product, and omega cross r
        """
        n = self._blade_plane(beta)
        r = self._airfoil_plane(beta)  # same as radius plane
        cross = np.cross(n, r)

        V_dot_n = np.einsum("ijk,ik->ij", self.V, n)
        V_dot_nxr = np.einsum("ijk,ik->ij", self.V, cross)
        omega_cross_r = self.omega * self.r.reshape(-1, 1) * np.cos(beta)
        V_airfoil_plane = ((V_dot_n[:, :, np.newaxis]) * (n[:, np.newaxis, :])) + (
            (V_dot_nxr + omega_cross_r.T)[:, :, np.newaxis] * cross[:, np.newaxis, :]
        )
        self.phi = np.arctan(-V_dot_n / (V_dot_nxr + omega_cross_r.T))
        V_airfoil_plane = (np.linalg.norm(V_airfoil_plane, axis=2)) ** 2
        self.mach_number = np.sqrt(V_airfoil_plane) / self.speed_of_sound

        Cl = self._Cl(self.linear_twist)
        Cd = self._Cd(Cl)
        dL = (
            0.5 * self.density * V_airfoil_plane * self.chord.reshape(1, -1) * Cl
        ) * self.dr
        dD = (
            0.5 * self.density * V_airfoil_plane * self.chord.reshape(1, -1) * Cd
        ) * self.dr
        dT = dL * np.cos(self.phi) - dD * np.sin(self.phi)
        dF_x = dL * np.sin(self.phi) + dD * np.cos(self.phi)
        return dT, dF_x, n, r, cross, omega_cross_r

    def _calculate_forward_flight_params(
        self,
    ):
        """
        Internal function to calculate the forward flight parameters

        Parameters:
        -----------
        None

        Returns:
        --------
        npt.NDArray
            The net thrust in forward flight
        npt.NDArray
            The power in forward flight
        npt.NDArray
            The torque in forward flight
        npt.NDArray
            The tip path plane in forward flight
        """
        dT, dF_x, n, r, _, omega_cross_r = self._forward_flight_blade_forces(self.beta)
        thrust = (np.sum(dT, axis=1) * n.T).T
        self.thrust = np.sum(thrust, axis=0)  # sum over all blades
        self.power = float(np.sum(dF_x * omega_cross_r.T))
        self.torque = np.sum(
            (np.sum(dF_x * self.r, axis=1) * (-n.T)).T,
            axis=0,
        )
        self.tpp = np.cross((r[0] - r[1]), (r[0] - r[2]))
        self.tpp /= np.linalg.norm(self.tpp)

    def _coning_angle(self) -> None:
        """
        Calculate the coning angle iteratively using fsolve

        Parameters:
        -----------
        None

        Returns:
        --------
        float
            The coning angle
        """

        def __objective_function(beta: npt.NDArray) -> npt.NDArray:
            dT, *_ = self._forward_flight_blade_forces(beta)
            dM = dT * self.r
            dF_centri = (
                self.omega**2
                * self.element_mass
                * np.outer(np.sin(beta) * np.cos(beta), self.r)
            )
            dM_centri = dF_centri * self.r
            M = np.sum(dM, axis=1)
            M_centri = np.sum(dM_centri, axis=1)
            return M_centri - M

        beta = np.deg2rad(np.ones_like(self.psi))
        self.beta = np.array(fsolve(__objective_function, beta))

    def _init_glauert_loop(self, norm_V_infty: float) -> npt.NDArray:
        """
        Initialize the Glauert loop. Start off by taking some induced velocity calculated by starting with 0 induced velocity and low forward velocity

        Parameters:
        -----------
        None

        Returns:
        --------
        V_i: npt.NDArray
            The induced velocity array, to be used in the Glauert loop
        """
        lambda_i_glauert = 0.03
        mu_lambda_G = (norm_V_infty / (self.omega * self.radius)) / 0.025
        lambda_i = lambda_i_glauert * (
            1
            + (
                (((4 / 3) * mu_lambda_G) / (1.2 + mu_lambda_G))
                * np.outer(np.cos(self.psi), self.r)
                / self.radius
            )
        )
        V_i = lambda_i * self.omega * self.radius

        tpp = np.array([0, 0, 1])
        V_i = (
            -V_i[:, :, np.newaxis] * tpp[np.newaxis, np.newaxis, :]
        )  # (NBLADES, NELEMENTS, (x, y, z))
        self.V = self.V_infty + V_i
        self._coning_angle()
        self._calculate_forward_flight_params()
        return V_i

    def _glauert_loop(
        self,
        ERROR: npt.NDArray,
        TOLERANCE: npt.NDArray,
        LIMIT: int,
        norm_V_infty: float,
        V_i: npt.NDArray,
    ) -> None:
        """
        Iteratively solve for the induced velocity, and converge thrust values

        Parameters:
        -----------
        ERROR: npt.NDArray
            The error in the thrust between iterations
        TOLERANCE: npt.NDArray
            The tolerance in the error of thrust between iterations -- used for convergence
        LIMIT: int
            The maximum number of iterations before breaking
        norm_V_infty: float
            The forward speed of the helicopter
        V_i: npt.NDArray
            The induced velocity array, taken from the initialization step

        Returns:
        --------
        None
        """

        def __objective_function(lambda_i_glauert, C_T, mu, alpha_tpp):
            return C_T - 2 * lambda_i_glauert * np.sqrt(
                mu**2 + (mu * np.tan(alpha_tpp) + lambda_i_glauert) ** 2
            )

        counter = 0

        while np.sum(ERROR > TOLERANCE) > 0:
            V_infty_dot_tpp = np.dot(self.V_infty, self.tpp)
            self.alpha_tpp = np.arccos(V_infty_dot_tpp / norm_V_infty) - np.pi / 2

            mu = (
                norm_V_infty * np.cos(self.alpha_tpp) / (self.omega * self.radius)
            )  # advance ratio

            net_thrust_mag = np.linalg.norm(self.thrust)
            C_T = net_thrust_mag / (
                self.density * self.area * (self.omega * self.radius) ** 2
            )
            lambda_i_glauert = fsolve(
                __objective_function, C_T / (2 * mu), args=(C_T, mu, self.alpha_tpp)
            )
            lambda_G = (
                -V_infty_dot_tpp / (self.omega * self.radius)
            ) + lambda_i_glauert
            mu_lambda_G = mu / lambda_G
            lambda_i = lambda_i_glauert * (
                1
                + (
                    (((4 / 3) * mu_lambda_G) / (1.2 + mu_lambda_G))
                    * np.outer(np.cos(self.psi), self.r)
                    / self.radius
                )
            )
            V_i_ = lambda_i * self.omega * self.radius
            V_i = (
                -V_i_[:, :, np.newaxis] * self.tpp[np.newaxis, np.newaxis, :] + V_i
            ) / 2  # (NBLADES, NELEMENTS, (x, y, z))
            self.V = self.V_infty + V_i
            self._coning_angle()
            ERROR = self.thrust
            TOLERANCE = np.abs(self.thrust * 0.05)
            self._calculate_forward_flight_params()
            ERROR = np.abs(ERROR - self.thrust)
            counter += 1
            if counter > LIMIT:
                display_text("Thrust did not converge")
                break

    def forward_flight_params(self):
        """
        Calculate the forward flight parameters by iteratively solving for
        induced velocity, and converging thrust values

        Parameters:
        -----------
        V_infty: npt.NDArray
            The forward velocity of the helicopter
        """
        norm_V_infty = float(np.linalg.norm(self.V_infty))
        if np.abs(self.V_infty[0]) > 0:
            V_i = self._init_glauert_loop(norm_V_infty)

            ERROR = np.inf * np.ones_like(self.thrust)
            TOLERANCE = np.array([0, 0, 0])
            LIMIT = 20

            self._glauert_loop(ERROR, TOLERANCE, LIMIT, norm_V_infty, V_i)

        else:
            ### Initialize induced velocity calc by passing it along [0, 0, 1] first
            lambda_i = self._ptl()  # (NBLADES, NELEMENTS)
            V_i = lambda_i * self.omega * self.radius
            tpp = np.array([0, 0, 1])
            V_i = (
                -V_i[:, :, np.newaxis] * tpp[np.newaxis, np.newaxis, :]
            )  # (NBLADES, NELEMENTS, (x, y, z))

            self.V = self.V_infty + V_i

            self._coning_angle()
            self._calculate_forward_flight_params()

            ### Now pass it along the actual tpp computed during the previous step
            ### This doesnt make much of a different to the output though
            V_i = lambda_i * self.omega * self.radius
            V_i = (
                -V_i[:, :, np.newaxis] * self.tpp[np.newaxis, np.newaxis, :]
            )  # (NBLADES, NELEMENTS, (x, y, z))
            self.V = self.V_infty + V_i

            self._coning_angle()
            self._calculate_forward_flight_params()
