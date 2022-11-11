import cv2
import torch

from methods import augment_images_and_map_correspondence


def render_correspondence_and_save(image: torch.Tensor, matches: torch.Tensor, filepath: str) -> None:
    """Example function written for a sigle image and not batch image"""

    image = image[0].permute(1, 2, 0).cpu().numpy()  # Making it channel last image
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # I -> [0, 255]

    matches = matches[0].cpu().numpy()

    for u, v in zip(matches[:, 0], matches[:, 1]):
        cv2.circle(image, (int(v), int(u)), radius=1, color=(255, 0, 255), thickness=2)
        """
        image = cv2.putText(image, f'{u, v}', (v + 5, u - 5), cv2.FONT_HERSHEY_COMPLEX,
                            0.3, (255, 0, 255), 1, cv2.LINE_AA)"""

    cv2.imwrite(filepath, image)


if __name__ == "__main__":

    read_image = cv2.imread("images/tsukuba-imL-300x225.png")

    # Init. torch.Tensor on CUDA as processing is faster!
    normalized_image = torch.as_tensor(read_image / 255, dtype=torch.float32, device=torch.device("cuda"))

    # Channel first setting (Capable of batch processing!) I (H x W x 3) -> I (B x 3 x H x W)
    batched_channel_first_image = normalized_image.unsqueeze(dim=0).permute(0, 3, 1, 2)

    image_a, matches_a, image_b, matches_b = augment_images_and_map_correspondence(batched_channel_first_image,
                                                                                   n_correspondence=50)

    # Render and save
    render_correspondence_and_save(image_a, matches_a, filepath="images/image_a.png")
    render_correspondence_and_save(image_b, matches_b, filepath="images/image_b.png")
