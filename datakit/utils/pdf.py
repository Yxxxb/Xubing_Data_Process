import io
import math
from typing import List

import fitz
import PyPDF2
from PIL import Image
from tqdm import tqdm


def split_pdf_into_sliced_pdfs(input_pdf_path: str,
                               output_folder: str,
                               slice_size: int = 1,
                               min_pages: int = 1,
                               max_pages: int = math.inf) -> None:
    """Split a pdf into multiple pdfs with a given size.

    Args:
        input_pdf_path (str): Path to the input pdf.
        output_folder (str): Path to the output folder.
        slice_size (int): Size of the slices. Defaults to 1.
        min_pages (int): Minimum number of pages.
            Defaults to 1.
        max_pages (int): Maximum number of pages.
            Defaults to math.inf.
    """
    with open(input_pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        # temporal choice
        num_pages = len(pdf_reader.pages) if max_pages > 10000 else max_pages
        for i in tqdm(range(min_pages, num_pages)):
            if i >= max_pages:
                break
            if i % slice_size == 0:
                pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[i])
            output_pdf_path = f'{output_folder}/{i//slice_size}.pdf'
            with open(output_pdf_path, 'wb') as output_pdf:
                pdf_writer.write(output_pdf)
    print(
        f'Split {input_pdf_path} into {(num_pages-min_pages)//slice_size} pdfs.'  # noqa
    )


def get_pdf_page_count(pdf_path: str) -> int:
    """Get the number of pages of a pdf.

    Args:
        pdf_path (str): Path to the pdf.

    Returns:
        int: Number of pages.
    """
    reader = PyPDF2.PdfReader(pdf_path)
    return len(reader.pages)


def extract_frames_from_pdf_and_save(pdf_path: str,
                                     first_page: int,
                                     last_page: int,
                                     output_folder: str = './',
                                     zoom_size: float = 2.0,
                                     save: bool = False) -> List[Image.Image]:
    """Extract frames from a pdf and save them as images.

    Args:
        pdf_path (str): Path to the pdf.
        first_page (int): First page to extract.
        last_page (int): Last page to extract.
        output_folder (str): Path to the output folder.
        zoom_size (float, optional): Zoom size. Defaults to 2.0.
        save (bool, optional): Whether to save the images.
            Defaults to False.
    """
    pdf_document = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom_size, zoom_size)
    images = []
    image_paths = []
    for i in range(first_page, last_page):
        page = pdf_document.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        image_bytes = pix.tobytes('png')
        image = Image.open(io.BytesIO(image_bytes))
        images.append(image)
        if save:
            tmp_path = f'{output_folder}/{i}.jpg'
            image.save(tmp_path)
            image_paths.append(tmp_path)
    if save:
        return image_paths
    else:
        return images
