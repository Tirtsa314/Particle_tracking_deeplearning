


def get_target_image(image_of_particles):
    """Get the target image from the image of particles."""

    label = np.zeros(image_of_particles.shape[:2]).astype(int)

    H, W = image_of_particles.shape[:2]
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    R = 30/np.sqrt(3)
    x, y = np.meshgrid(
        np.arange(0, image_of_particles.shape[0]),
        np.arange(0, image_of_particles.shape[1]),
    )

    for property in image_of_particles.properties:
      if "position" in property:
            #position = property["position"]
            r0, c0 = property["position"]
            #rotation = property["rotation"][0]
            rot = property.get("rotation", (0.0, 0.0, 0.0))

            # Rotate the grid
            #theta = rotation  # single angle in radians
            theta = float(rot[0]) if np.ndim(rot) != 0 else float(rot)

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # XR = cos_t * x - sin_t * y
            # YR = sin_t * x + cos_t * y
            X = cc - c0
            Y = rr - r0

            XR = cos_t * X - sin_t * Y
            YR = sin_t * X + cos_t * Y
            # Triangle mask
            mask = (
            (-R/2 < XR) & (XR < R) &
            ((np.sqrt(3)/3 * XR - np.sqrt(3)/3 * R) < YR) &
            (YR < (-np.sqrt(3)/3 * XR + np.sqrt(3)/3 * R))
    )
            label[mask] = (1)
    return label