from lab2.utils.consts import BETA, ALPHA, M, SIGMA, KEY
from lab2.utils.embedding import additional_embedding
from lab2.utils.fourier import get_fft_image, get_abs_matrix, get_phase_matrix, get_complex_matrix, \
    get_inverse_fft_image
from lab2.utils.in_out import read_image, write_image
from lab2.utils.snipping import get_H_zone, merge_pictures_H_zone
from lab2.utils.watermark import generate_watermark


def do_embedding(image_path, result_path):
    container = read_image(image_path)

    # 1. Реализовать генерацию ЦВЗ 𝛺 как псевдослучайной последовательности заданной длины из чисел,
    # распределённых по нормальному закону
    H_zone_length = int(container.shape[0] * 0.5) * int(container.shape[1] * 0.5)
    watermark, _ = generate_watermark(H_zone_length, M, SIGMA, KEY)

    # 2. Реализовать трансформацию исходного контейнера к пространству признаков
    fft_container = get_fft_image(container)
    abs_fft_container = get_abs_matrix(fft_container)
    phase_fft_container = get_phase_matrix(fft_container)

    # 3. Осуществить встраивание информации аддитивным методом встраивания.
    # Значения параметравстраивания устанавливается произвольным образом.
    H_zone = get_H_zone(abs_fft_container)
    watermark = watermark.reshape(H_zone.shape)
    H_zone_watermark = additional_embedding(H_zone, BETA, watermark, ALPHA)

    # 4. Сформировать носитель информации при помощи обратного преобразования
    # от матрицы признаков к цифровому сигналу.  Сохранить его на диск.
    merged_abs_picture = merge_pictures_H_zone(abs_fft_container, H_zone_watermark)
    complex_matrix = get_complex_matrix(merged_abs_picture, phase_fft_container)
    processed_image = get_inverse_fft_image(complex_matrix)
    write_image(processed_image, result_path)
