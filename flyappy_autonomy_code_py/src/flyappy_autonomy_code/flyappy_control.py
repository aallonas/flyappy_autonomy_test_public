class FlyappyControl:

    def __init__(self):
        self.kp = 4.0      # position -> velocity
        self.kd = 10.0     # velocity -> acceleration

        self.v_max = 8.0   # velocity limit
        self.a_max = 35.0  # acceleration limit

    def pid_controller(self, setpoint, value, kp, ki, kd, previous_error, integral, dt):
        error = setpoint - value
        integral += error * dt
        derivative = (error - previous_error) / dt
        control = kp * error + ki * integral + kd * derivative
        return control, error, integral
    
    def state_feedback_controller(self,
                                  value: float,
                                  velocity: float,
                                  setpoint: float,
                                  ) -> float:
        pos_error = setpoint - value
        v_des = self.kp * pos_error

        # Target velocity
        v_des = max(min(v_des, self.v_max), -self.v_max)

        # Computed acceleration
        a = self.kd * (v_des - velocity)

        # acceleration clamp
        a = max(min(a, self.a_max), -self.a_max)

        return a
                                
                                  




