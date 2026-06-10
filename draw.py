import cv2
import numpy as np
import os

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


def draw_hand(canvas, hand_data, color=(255, 255, 255)):
    """
    hand_data shape = (21, 3)
    """

    points = []

    for x, y, z in hand_data:

        if x == 0 and y == 0 and z == 0:
            points.append(None)
            continue

        px = int(x * 500 + 50)
        py = int(y * 500 + 50)

        points.append((px, py))

        cv2.circle(canvas, (px, py), 4, color, -1)

    for start, end in HAND_CONNECTIONS:

        if (
            points[start] is not None and
            points[end] is not None
        ):
            cv2.line(
                canvas,
                points[start],
                points[end],
                color,
                2
            )


while True:

    choice = input(
        "\nEnter word name or chunk number ('q' to quit): "
    ).strip()

    if choice.lower() == "q":
        break

    data = None

    # Try word dataset
    word_path = (
        f"dataset_landmarks/word_chunks_coord/{choice}.npy"
    )

    if os.path.exists(word_path):

        print("Word found")
        data = np.load(word_path)

    else:

        # Try alphabet chunk
        try:

            alpha_path = (
                f"dataset_landmarks/alphabets_chunks_coord/chunk_{int(choice)}.npy"
            )

            if os.path.exists(alpha_path):

                print("Alphabet found")
                data = np.load(alpha_path)

        except ValueError:
            pass

    if data is None:

        print("File not found")
        continue

    print("Shape:", data.shape)

    stop = False

    for frame_idx, frame in enumerate(data):

        canvas = np.zeros((600, 600, 3), dtype=np.uint8)

        # Single hand dataset
        if len(frame) == 63:

            hand = frame.reshape(21, 3)

            draw_hand(
                canvas,
                hand,
                (255, 255, 255)
            )

        # Two hand dataset
        elif len(frame) == 126:

            left_hand = frame[:63].reshape(21, 3)
            right_hand = frame[63:].reshape(21, 3)

            draw_hand(
                canvas,
                left_hand,
                (255, 255, 255)
            )

            draw_hand(
                canvas,
                right_hand,
                (0, 255, 255)
            )

        # cv2.putText(
        #     canvas,
        #     f"Frame: {frame_idx + 1}/{len(data)}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 0),
        #     2
        # )

        cv2.imshow("Skeleton Replay", canvas)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            stop = True
            break

    cv2.destroyAllWindows()

    if stop:
        print("Replay stopped")

cv2.destroyAllWindows()