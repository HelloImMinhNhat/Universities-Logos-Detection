import cv2
import os
import numpy as np

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Đường dẫn đầy đủ của tệp hình ảnh đầu vào
            input_path = os.path.join(input_folder, filename)

            # Đọc hình ảnh từ tệp vào biến img
            img = cv2.imread(input_path)

            # Xoay hình ảnh ngang
            rotated_horizontal = cv2.flip(img, 1)

            # Xoay hình ảnh dọc
            rotated_vertical = cv2.flip(img, 0)

            # Lật hình ảnh sang trái
            flipped_left = cv2.flip(img, -1)

            # Lật hình ảnh sang phải
            flipped_right = img

            # Lưu hình ảnh sau khi xử lý vào thư mục đầu ra
            cv2.imwrite(os.path.join(output_folder, 'rotated_horizontal_' + filename), rotated_horizontal)
            cv2.imwrite(os.path.join(output_folder, 'rotated_vertical_' + filename), rotated_vertical)
            cv2.imwrite(os.path.join(output_folder, 'flipped_left_' + filename), flipped_left)
            cv2.imwrite(os.path.join(output_folder, 'flipped_right_' + filename), flipped_right)

def rename_files_in_folder(folder_path, start_index=0):
    # Kiểm tra xem đường dẫn thư mục có tồn tại không
    if not os.path.exists(folder_path):
        print(f"Thư mục '{folder_path}' không tồn tại.")
        return

    # Lấy danh sách tất cả các tệp trong thư mục
    files = os.listdir(folder_path)

    # Lặp qua từng tệp và đổi tên
    for index, file_name in enumerate(files):
        # Lấy đuôi mở rộng của tệp
        file_extension = os.path.splitext(file_name)[1]

        # Tạo tên mới cho tệp
        new_index = start_index + index
        new_file_name = f"{new_index}{file_extension}"

        # Đường dẫn đầy đủ của tệp cũ và mới
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Kiểm tra xem tệp mới đã tồn tại chưa
        while os.path.exists(new_file_path):
            new_index += 1
            new_file_name = f"{new_index}{file_extension}"
            new_file_path = os.path.join(folder_path, new_file_name)

        # Đổi tên tệp
        os.rename(old_file_path, new_file_path)

def augment_images(input_folder, output_folder):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lặp qua tất cả các tập tin trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Đường dẫn đầy đủ của tệp hình ảnh đầu vào
            input_path = os.path.join(input_folder, filename)

            # Đọc hình ảnh từ tệp vào biến img
            img = cv2.imread(input_path)

            # Áp dụng các biến thay đổi khác nhau
            augmented_images = []
            for i in range(5):  # Số lượng hình ảnh được tạo ra cho mỗi hình ảnh đầu vào
                # Xoay hình ảnh ngẫu nhiên
                rotated_image = cv2.rotate(img, np.random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]))

                # Áp dụng độ sáng ngẫu nhiên
                brightness_factor = np.random.uniform(0.7, 1.3)
                brightened_image = cv2.convertScaleAbs(rotated_image, alpha=brightness_factor, beta=0)

                # Áp dụng độ tương phản ngẫu nhiên
                contrast_factor = np.random.uniform(0.8, 1.2)
                contrasted_image = cv2.convertScaleAbs(brightened_image, alpha=contrast_factor, beta=0)

                # Áp dụng màu sắc ngẫu nhiên
                hue_shift = np.random.randint(-10, 10)
                sat_shift = np.random.uniform(0.8, 1.2)
                value_shift = np.random.uniform(0.8, 1.2)
                hsv_image = cv2.cvtColor(contrasted_image, cv2.COLOR_BGR2HSV)
                hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
                hsv_image[:, :, 1] = np.clip(sat_shift * hsv_image[:, :, 1], 0, 255)
                hsv_image[:, :, 2] = np.clip(value_shift * hsv_image[:, :, 2], 0, 255)
                colored_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

                # Lưu hình ảnh sau khi xử lý vào thư mục đầu ra
                output_filename = f"{i + 1}_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, colored_image)

if __name__ == "__main__":
    # Thay đổi đường dẫn thư mục đầu vào và đầu ra tùy thuộc vào nhu cầu của bạn
    input_folder = "./data/train/USSH"

    output_folder = "./data/QNU/"

    # process_images(input_folder, output_folder)
    rename_files_in_folder(input_folder, start_index=2322)

    # augment_images(input_folder, output_folder)