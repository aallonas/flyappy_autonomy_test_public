from typing import Tuple
import numpy as np

"""" control algorithms for the Flyappy autonomous agent """

class ProportionalController:

    def __init__(
            self,
            kp: float = 4.0,
            acc_max: float = 3.0, 
            ) -> None:
        # Controller gains
        self.kp = kp     
        self.acc_max = acc_max

    def control_step(
            self,
            setpoint : float,
            velocity: float,
            dt: float
            ) -> float:
        """ Proportional controller 
        
        Args:
            setpoint (float): Desired velocity [m/s]
            velocity (float): Current velocity [m/s]
        
        Returns:
            acceleration (float): Control output [m/s²]
            error (float): Current velocity error [m/s]
        """
        # Compute error
        error = setpoint - velocity
        acceleration = self.kp * error 
        # Enforce acceleration limits
        acceleration = np.clip(acceleration, -self.acc_max, self.acc_max)
        return acceleration
    
    def reset(self) -> None:
        pass


class StateFeedbackController:

    def __init__(
            self,
            kp: float = 4.0,
            kd: float = 10.0,
            vel_max: float = 8.0,
            acc_max: float = 35.0,
            ) -> None:
        # Controller gains
        self.kp = kp     
        self.kd = kd
        # Limits
        self.vel_max = vel_max
        self.acc_max = acc_max

    def control_step(
            self,
            setpoint: float,
            position: float,
            velocity: float
            ) -> float:
        """ State feedback controller

        Uses a two step approach:
        1. Position error -> desired velocity
            - Proportional controller
            - Enforce velocity limits
        2. Velocity error -> acceleration command
            - Derivative controller
            - Enforce acceleration limits

        Args:
            position (float): Current position [m]
            velocity (float): Current velocity [m/s]
            setpoint (float): Desired position [m]

        Returns:
            acceleration (float): Acceleration command [m/s²]
        """

        # Step 1: Position error to desired velocity
        pos_error = setpoint - position
        vel_des = self.kp * pos_error
        vel_des = np.clip(vel_des, -self.vel_max, self.vel_max)

        # Step 2: Velocity error to acceleration
        acceleration = self.kd * (vel_des - velocity)
        acceleration = np.clip(acceleration, -self.acc_max, self.acc_max)
        return float(acceleration)
    
    def reset(self) -> None:
        pass