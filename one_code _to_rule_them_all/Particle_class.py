import numpy as np

def angle_difference_triangle(theta_new, theta_old):
    """
    Smallest angular difference for a 3-fold symmetric triangle.
    Angles in radians.
    Returns difference in range [-pi/3, pi/3].
    """
    period = 2 * np.pi / 3
    dtheta = theta_new - theta_old

    dtheta = (dtheta + period / 2) % period - period / 2

    return dtheta




class ParticleTrack:
    def __init__(self, particle_id, x, y, theta, frame):
        self.id = particle_id
        self.x = x
        self.y = y
        self.theta = theta

        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0

        self.last_frame = frame
        self.missed = 0
        self.history = []

    def predict(self, frame, dt=1):
        x_pred = self.x + self.vx * dt
        y_pred = self.y + self.vy * dt
        theta_pred = self.theta + self.omega * dt
        return x_pred, y_pred, theta_pred

    def update(self, detection, frame, dt=1):
        x_new = detection["x"]
        y_new = detection["y"]
        theta_new = detection["theta"]

        self.vx = (x_new - self.x) / dt
        self.vy = (y_new - self.y) / dt
        self.omega = angle_difference_triangle(theta_new, self.theta) / dt

        self.x = x_new
        self.y = y_new
        self.theta = theta_new
        self.last_frame = frame
        self.missed = 0

        self.history.append({
            "frame": frame,
            "particle": self.id,
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "vx": self.vx,
            "vy": self.vy,
            "omega": self.omega,
            "source": detection.get("source", "full_model"),
        })

    def mark_missing(self, frame):
        self.missed += 1
        x_pred, y_pred, theta_pred = self.predict(frame)

        self.history.append({
            "frame": frame,
            "particle": self.id,
            "x": x_pred,
            "y": y_pred,
            "theta": theta_pred,
            "vx": self.vx,
            "vy": self.vy,
            "omega": self.omega,
            "source": "predicted_missing",
        })