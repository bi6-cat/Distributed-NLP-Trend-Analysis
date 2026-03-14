"""
Task 1.1 — Test VnCoreNLP chạy được local
"""
import vncorenlp
import os

VNCORENLP_JAR = os.path.abspath("vncorenlp/VnCoreNLP-1.1.1.jar")

def test_word_segmentation():
    annotator = vncorenlp.VnCoreNLP(
        VNCORENLP_JAR,
        annotators="wseg",
        max_heap_size="-Xmx512m"
    )

    test_sentences = [
        "Điện thoại iPhone 15 bị lỗi màn hình nghiêm trọng.",
        "VinFast ra mắt xe điện mới tại thị trường Mỹ.",
        "ChatGPT đang thay thế lập trình viên hay không?",
    ]

    print("=== VnCoreNLP Word Segmentation Test ===\n")
    for sent in test_sentences:
        result = annotator.tokenize(sent)
        print(f"Input : {sent}")
        print(f"Output: {result}\n")

    annotator.close()
    print("✅ VnCoreNLP hoạt động bình thường!")

if __name__ == "__main__":
    test_word_segmentation()