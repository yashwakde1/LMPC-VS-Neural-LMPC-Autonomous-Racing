import numpy as np
import numpy.linalg as la
import pdb

class Map():
    """Custom oval track map object
    Attributes:
        getGlobalPosition: convert position from (s, ey) to (X,Y)
        getLocalPosition: convert position from (X, Y) to (s, ey)
        curvature: get curvature at position s
    """
    def __init__(self, halfWidth=0.5):
        """Initialization
        halfWidth: track halfWidth
        """
        self.halfWidth = halfWidth
        self.slack = 0.4

        # ===============================
        # CUSTOM OVAL TRACK SPECIFICATION
        # ===============================
        # Format: [segment_length, radius]
        # Positive radius = left turn, negative = right turn, 0 = straight
        R = 5.0          # curve radius (m)
        straight = 8.0   # straight length (m)
        
        spec = np.array([
            [straight, 0.0],           # First straight
            [np.pi * R / 2, R],        # Quarter circle (left turn)
            [straight, 0.0],           # Second straight
            [np.pi * R / 2, R]         # Quarter circle (left turn) - closes the oval
        ])

        # Now compute (x, y) points and tangent angles
        # PointAndTangent = [x, y, psi, cumulative s, segment length, signed curvature]
        PointAndTangent = np.zeros((spec.shape[0] + 1, 6))
        
        for i in range(0, spec.shape[0]):
            if spec[i, 1] == 0.0:  # Straight line segment
                l = spec[i, 0]
                if i == 0:
                    ang = 0
                    x = 0 + l * np.cos(ang)
                    y = 0 + l * np.sin(ang)
                else:
                    ang = PointAndTangent[i - 1, 2]
                    x = PointAndTangent[i-1, 0] + l * np.cos(ang)
                    y = PointAndTangent[i-1, 1] + l * np.sin(ang)
                psi = ang

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 0])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 0])

                PointAndTangent[i, :] = NewLine
                
            else:  # Curved segment
                l = spec[i, 0]
                r = spec[i, 1]

                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                if i == 0:
                    ang = 0
                    CenterX = 0 + np.abs(r) * np.cos(ang + direction * np.pi / 2)
                    CenterY = 0 + np.abs(r) * np.sin(ang + direction * np.pi / 2)
                else:
                    ang = PointAndTangent[i - 1, 2]
                    CenterX = PointAndTangent[i-1, 0] + np.abs(r) * np.cos(ang + direction * np.pi / 2)
                    CenterY = PointAndTangent[i-1, 1] + np.abs(r) * np.sin(ang + direction * np.pi / 2)

                spanAng = l / np.abs(r)
                psi = wrap(ang + spanAng * np.sign(r))

                angleNormal = wrap((direction * np.pi / 2 + ang))
                angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))
                x = CenterX + np.abs(r) * np.cos(angle + direction * spanAng)
                y = CenterY + np.abs(r) * np.sin(angle + direction * spanAng)

                if i == 0:
                    NewLine = np.array([x, y, psi, PointAndTangent[i, 3], l, 1 / r])
                else:
                    NewLine = np.array([x, y, psi, PointAndTangent[i-1, 3] + PointAndTangent[i-1, 4], l, 1 / r])

                PointAndTangent[i, :] = NewLine

        # Close the loop - connect last point back to origin
        xs = PointAndTangent[-2, 0]
        ys = PointAndTangent[-2, 1]
        xf = 0
        yf = 0
        psif = 0

        l = np.sqrt((xf - xs) ** 2 + (yf - ys) ** 2)
        NewLine = np.array([xf, yf, psif, PointAndTangent[-2, 3] + PointAndTangent[-2, 4], l, 0])
        PointAndTangent[-1, :] = NewLine

        self.PointAndTangent = PointAndTangent
        self.TrackLength = PointAndTangent[-1, 3] + PointAndTangent[-1, 4]

    def getGlobalPosition(self, s, ey):
        """coordinate transformation from curvilinear reference frame (s, ey) to inertial reference frame (X, Y)
        (s, ey): position in the curvilinear reference frame
        """
        # wrap s along the track
        while (s > self.TrackLength):
            s = s - self.TrackLength

        # Compute the segment in which system is evolving
        PointAndTangent = self.PointAndTangent

        index = np.all([[s >= PointAndTangent[:, 3]], [s < PointAndTangent[:, 3] + PointAndTangent[:, 4]]], axis=0)
        i = int(np.where(np.squeeze(index))[0])

        if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
            xf = PointAndTangent[i, 0]
            yf = PointAndTangent[i, 1]
            xs = PointAndTangent[i - 1, 0]
            ys = PointAndTangent[i - 1, 1]
            psi = PointAndTangent[i, 2]

            deltaL = PointAndTangent[i, 4]
            reltaL = s - PointAndTangent[i, 3]

            x = (1 - reltaL / deltaL) * xs + reltaL / deltaL * xf + ey * np.cos(psi + np.pi / 2)
            y = (1 - reltaL / deltaL) * ys + reltaL / deltaL * yf + ey * np.sin(psi + np.pi / 2)
        else:
            r = 1 / PointAndTangent[i, 5]
            ang = PointAndTangent[i - 1, 2]
            
            if r >= 0:
                direction = 1
            else:
                direction = -1

            CenterX = PointAndTangent[i - 1, 0] + np.abs(r) * np.cos(ang + direction * np.pi / 2)
            CenterY = PointAndTangent[i - 1, 1] + np.abs(r) * np.sin(ang + direction * np.pi / 2)

            spanAng = (s - PointAndTangent[i, 3]) / (np.pi * np.abs(r)) * np.pi

            angleNormal = wrap((direction * np.pi / 2 + ang))
            angle = -(np.pi - np.abs(angleNormal)) * (sign(angleNormal))

            x = CenterX + (np.abs(r) - direction * ey) * np.cos(angle + direction * spanAng)
            y = CenterY + (np.abs(r) - direction * ey) * np.sin(angle + direction * spanAng)

        return x, y

    def getLocalPosition(self, x, y, psi):
        """coordinate transformation from inertial reference frame (X, Y) to curvilinear reference frame (s, ey)
        (X, Y): position in the inertial reference frame
        """
        PointAndTangent = self.PointAndTangent
        CompletedFlag = 0

        for i in range(0, PointAndTangent.shape[0]):
            if CompletedFlag == 1:
                break

            if PointAndTangent[i, 5] == 0.0:  # If segment is a straight line
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                psi_unwrap = np.unwrap([PointAndTangent[i - 1, 2], psi])[1]
                epsi = psi_unwrap - PointAndTangent[i - 1, 2]
                
                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    s  = PointAndTangent[i, 3]
                    ey = 0
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    CompletedFlag = 1
                else:
                    if np.abs(computeAngle([x,y], [xs, ys], [xf, yf])) <= np.pi/2 and np.abs(computeAngle([x,y], [xf, yf], [xs, ys])) <= np.pi/2:
                        v1 = np.array([x,y]) - np.array([xs, ys])
                        angle = computeAngle([xf,yf], [xs, ys], [x, y])
                        s_local = la.norm(v1) * np.cos(angle)
                        s       = s_local + PointAndTangent[i, 3]
                        ey      = la.norm(v1) * np.sin(angle)

                        if np.abs(ey) <= self.halfWidth + self.slack:
                            CompletedFlag = 1
            else:
                xf = PointAndTangent[i, 0]
                yf = PointAndTangent[i, 1]
                xs = PointAndTangent[i - 1, 0]
                ys = PointAndTangent[i - 1, 1]

                r = 1 / PointAndTangent[i, 5]
                if r >= 0:
                    direction = 1
                else:
                    direction = -1

                ang = PointAndTangent[i - 1, 2]

                CenterX = xs + np.abs(r) * np.cos(ang + direction * np.pi / 2)
                CenterY = ys + np.abs(r) * np.sin(ang + direction * np.pi / 2)

                if (la.norm(np.array([xs, ys]) - np.array([x, y]))) == 0:
                    ey = 0
                    psi_unwrap = np.unwrap([ang, psi])[1]
                    epsi = psi_unwrap - ang
                    s = PointAndTangent[i, 3]
                    CompletedFlag = 1
                elif (la.norm(np.array([xf, yf]) - np.array([x, y]))) == 0:
                    s = PointAndTangent[i, 3] + PointAndTangent[i, 4]
                    ey = 0
                    psi_unwrap = np.unwrap([PointAndTangent[i, 2], psi])[1]
                    epsi = psi_unwrap - PointAndTangent[i, 2]
                    CompletedFlag = 1
                else:
                    arc1 = PointAndTangent[i, 4] * PointAndTangent[i, 5]
                    arc2 = computeAngle([xs, ys], [CenterX, CenterY], [x, y])
                    if np.sign(arc1) == np.sign(arc2) and np.abs(arc1) >= np.abs(arc2):
                        v = np.array([x, y]) - np.array([CenterX, CenterY])
                        s_local = np.abs(arc2)*np.abs(r)
                        s    = s_local + PointAndTangent[i, 3]
                        ey   = -np.sign(direction) * (la.norm(v) - np.abs(r))
                        psi_unwrap = np.unwrap([ang + arc2, psi])[1]
                        epsi = psi_unwrap - (ang + arc2)

                        if np.abs(ey) <= self.halfWidth + self.slack:
                            CompletedFlag = 1

        if epsi > 1.0:
            pdb.set_trace()

        if CompletedFlag == 0:
            s    = 10000
            ey   = 10000
            epsi = 10000
            print("Error!! POINT OUT OF THE TRACK!!!! <==================")
            pdb.set_trace()

        return s, ey, epsi, CompletedFlag

    def curvature(self, s):
        """curvature computation
        s: curvilinear abscissa at which the curvature has to be evaluated
        """
        TrackLength = self.PointAndTangent[-1,3]+self.PointAndTangent[-1,4]

        while (s > TrackLength):
            s = s - TrackLength

        index = np.all([[s >= self.PointAndTangent[:, 3]], [s < self.PointAndTangent[:, 3] + self.PointAndTangent[:, 4]]], axis=0)
        i = int(np.where(np.squeeze(index))[0])
        curvature = self.PointAndTangent[i, 5]

        return curvature

    def getAngle(self, s, epsi):
        """Get angle at position s with heading error epsi"""
        TrackLength = self.PointAndTangent[-1,3]+self.PointAndTangent[-1,4]

        while (s > TrackLength):
            s = s - TrackLength

        index = np.all([[s >= self.PointAndTangent[:, 3]], [s < self.PointAndTangent[:, 3] + self.PointAndTangent[:, 4]]], axis=0)
        i = int(np.where(np.squeeze(index))[0])

        if i > 0:
            ang = self.PointAndTangent[i - 1, 2]
        else:
            ang = 0

        if self.PointAndTangent[i, 5] == 0:
            r = 0
        else:
            r = 1 / self.PointAndTangent[i, 5]

        if r == 0:
            angle_at_s = ang + epsi
        else:
            cumulative_s = self.PointAndTangent[i, 3]
            relative_s = s - cumulative_s
            spanAng = relative_s / np.abs(r)
            psi = wrap(ang + spanAng * np.sign(r))
            angle_at_s = psi + epsi

        return angle_at_s


# ======================================================================================================================
# ====================================== Internal utilities functions ==================================================
# ======================================================================================================================
def computeAngle(point1, origin, point2):
    """Compute angle between vectors"""
    v1 = np.array(point1) - np.array(origin)
    v2 = np.array(point2) - np.array(origin)

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = np.arctan2(det, dot)

    return angle

def wrap(angle):
    """Wrap angle to [-pi, pi]"""
    if angle < -np.pi:
        w_angle = 2 * np.pi + angle
    elif angle > np.pi:
        w_angle = angle - 2 * np.pi
    else:
        w_angle = angle
    return w_angle

def sign(a):
    """Sign function"""
    if a >= 0:
        res = 1
    else:
        res = -1
    return res