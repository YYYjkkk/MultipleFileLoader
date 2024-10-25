from typing import List
import cv2
import numpy as np
import tqdm
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from PIL import Image
from ocr import get_ocr  # 保留这个OCR相关的导入
import fitz  # PyMuPDF

class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            """Rotate the image by a given angle."""
            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            return cv2.warpAffine(img, M, (new_w, new_h))

        def is_dual_column(image):
            """Determine if an image has dual-column layout by analyzing pixel intensity distribution."""
            h, w = image.shape[:2]
            left_intensity = np.sum(image[:, :w // 2])  # Left half intensity
            right_intensity = np.sum(image[:, w // 2:])  # Right half intensity
            intensity_ratio = min(left_intensity, right_intensity) / max(left_intensity, right_intensity)
            return intensity_ratio > 0.7  # Threshold to determine if it's dual-column

        def pdf2text(filepath):
            """Convert PDF to text with OCR, supporting both single and dual-column formats."""
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""
            b_unit = tqdm.tqdm(total=doc.page_count, desc="Processing PDF pages")
            for i, page in enumerate(doc):
                b_unit.set_description(f"Page {i+1}/{doc.page_count}")
                b_unit.refresh()
                img_list = page.get_image_info(xrefs=True)
                resp += page.get_text("") + "\n"

                for img in img_list:
                    if xref := img.get("xref"):
                        pix = fitz.Pixmap(doc, xref)
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                        if page.rotation != 0:
                            ori_img = cv2.cvtColor(np.array(Image.fromarray(img_array)), cv2.COLOR_RGB2BGR)
                            img_array = cv2.cvtColor(rotate_img(ori_img, 360 - page.rotation), cv2.COLOR_BGR2RGB)

                        if is_dual_column(img_array):
                            h, w, _ = img_array.shape
                            left_img = img_array[:, :w // 2]  # Left column
                            right_img = img_array[:, w // 2:]  # Right column
                            # Perform OCR on both sections and merge results
                            for img_section in [left_img, right_img]:
                                result, _ = ocr(img_section)
                                if result:
                                    ocr_result = [line[1] for line in result]
                                    resp += "\n".join(ocr_result) + "\n"
                        else:
                            result, _ = ocr(img_array)
                            if result:
                                ocr_result = [line[1] for line in result]
                                resp += "\n".join(ocr_result)

                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)

    def save_to_txt(self, output_txt_path: str):
        """Saves the OCR output from the PDF to a text file."""
        docs = self.load()  # Load the documents
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc.page_content + '\n')  # Write the text of each document to the text file

if __name__ == "__main__":
    loader = RapidOCRPDFLoader(file_path="document_loaders/input/testForpdf_2cols.pdf")
    output_txt_path = "document_loaders/output/pdf2txt_output.txt"
    loader.save_to_txt(output_txt_path)
    print(f"OCR result saved to {output_txt_path}")
