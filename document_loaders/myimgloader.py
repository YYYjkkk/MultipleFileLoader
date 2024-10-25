from typing import List

from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

from ocr import get_ocr


class RapidOCRLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def img2text(filepath):
            resp = ""
            ocr = get_ocr()
            result, _ = ocr(filepath)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)
            return resp

        text = img2text(self.file_path)
        from unstructured.partition.text import partition_text

        return partition_text(text=text, **self.unstructured_kwargs)

    def save_to_txt(self, output_txt_path: str):
        """Saves the OCR output to a text file."""
        docs = self.load()  # Load the documents
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(doc.page_content + '\n')  # Write the text of each document to the text file


if __name__ == "__main__":
    loader = RapidOCRLoader(file_path="document_loaders/input/testForimg.jpg")
    output_txt_path = "document_loaders/output/jpg2txt_output.txt"  # Specify your output file path
    loader.save_to_txt(output_txt_path)
    print(f"OCR result saved to {output_txt_path}")
