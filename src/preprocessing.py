"""
Vietnamese Text Preprocessing Module
Chuyên xử lý văn bản tiếng Việt cho bài toán phân loại sản phẩm
"""

import re
import unicodedata
from typing import List, Optional, Callable
import pandas as pd

# Vietnamese stopwords - danh sách từ dừng phổ biến
VIETNAMESE_STOPWORDS = {
    # Đại từ nhân xưng
    'tôi', 'tao', 'mình', 'chúng_tôi', 'chúng_ta', 'bạn', 'các_bạn', 'anh', 'chị', 'em',
    'ông', 'bà', 'cô', 'chú', 'nó', 'họ', 'ai', 'gì',
    
    # Từ nối, liên từ
    'và', 'hoặc', 'hay', 'nhưng', 'mà', 'nên', 'vì', 'do', 'bởi', 'tuy', 'dù', 'nếu',
    'khi', 'lúc', 'trong_khi', 'sau_khi', 'trước_khi',
    
    # Giới từ  
    'của', 'cho', 'với', 'từ', 'đến', 'trong', 'ngoài', 'trên', 'dưới', 'giữa',
    'về', 'theo', 'bằng', 'qua', 'tại', 'vào', 'ra',
    
    # Trạng từ
    'rất', 'quá', 'lắm', 'cực', 'vô_cùng', 'hơi', 'khá', 'tương_đối',
    'đã', 'đang', 'sẽ', 'vẫn', 'còn', 'mới', 'vừa', 'luôn', 'thường',
    'cũng', 'chỉ', 'ngay', 'liền', 'mãi',
    
    # Đại từ chỉ định
    'này', 'kia', 'đó', 'ấy', 'đây', 'đấy', 'nọ',
    
    # Từ phủ định
    'không', 'chưa', 'chẳng', 'đừng', 'hãy',
    
    # Số từ
    'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'trăm', 'nghìn', 'ngàn', 'triệu', 'tỷ',
    
    # Từ chỉ lượng
    'nhiều', 'ít', 'mỗi', 'từng', 'mọi', 'tất_cả', 'toàn_bộ', 'hết',
    'các', 'những', 'một_số', 'vài', 'đủ',
    
    # Từ hỏi
    'sao', 'nào', 'đâu', 'bao_giờ', 'bao_nhiêu', 'bao_lâu',
    
    # Từ so sánh
    'hơn', 'nhất', 'như', 'bằng', 'kém', 'thua',
    
    # Từ đệm, từ phụ
    'thì', 'là', 'được', 'bị', 'có', 'làm', 'đi', 'lại', 'ra', 'vào',
    'lên', 'xuống', 'về', 'đến', 'tới', 'rồi', 'xong', 'hết', 'xin',
    'ạ', 'nhé', 'nha', 'hen', 'ha', 'hả', 'chứ', 'à', 'ơi', 'ư',
    
    # Từ chỉ thời gian
    'hôm_nay', 'hôm_qua', 'ngày_mai', 'tuần', 'tháng', 'năm', 'giờ', 'phút', 'giây',
    
    # Từ thường gặp trong mô tả sản phẩm
    'sản_phẩm', 'hàng', 'shop', 'giá', 'giá_sản_phẩm', 'tiki', 'bao_gồm', 'thuế',
    'luật', 'hiện_hành', 'bên_cạnh', 'tuỳ', 'loại', 'hình_thức', 'địa_chỉ',
    'giao_hàng', 'phát_sinh', 'thêm', 'chi_phí', 'khác', 'phí_vận_chuyển',
    'phụ_phí', 'hàng_cồng_kềnh', 'thuế_nhập_khẩu', 'đối_với', 'đơn_hàng',
    'giao', 'nước_ngoài', 'giá_trị', 'triệu_đồng', 'tài_sản', 'cá_nhân',
    'bán', 'nhà_bán_hàng', 'không_thuộc', 'đối_tượng', 'phải_chịu',
    'gtgt', 'hoá_đơn', 'vat', 'cung_cấp', 'trường_hợp'
}


class VietnameseTextPreprocessor:
    """
    Class xử lý văn bản tiếng Việt cho NLP
    """
    
    def __init__(
        self, 
        use_word_segmentation: bool = True,
        remove_stopwords: bool = True,
        custom_stopwords: Optional[set] = None
    ):
        """
        Khởi tạo bộ tiền xử lý
        
        Args:
            use_word_segmentation: Có sử dụng tách từ tiếng Việt không
            remove_stopwords: Có loại bỏ stopwords không  
            custom_stopwords: Danh sách stopwords tùy chỉnh
        """
        self.use_word_segmentation = use_word_segmentation
        self.remove_stopwords = remove_stopwords
        self.stopwords = VIETNAMESE_STOPWORDS.copy()
        
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Load word segmenter
        self._word_segment = None
        if use_word_segmentation:
            try:
                from underthesea import word_tokenize
                self._word_segment = word_tokenize
            except ImportError:
                print("Warning: underthesea not installed. Word segmentation disabled.")
                self.use_word_segmentation = False
    
    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa Unicode về dạng NFC"""
        return unicodedata.normalize('NFC', text)
    
    def remove_html_tags(self, text: str) -> str:
        """Loại bỏ HTML tags"""
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(['style', 'script', 'head', 'title', 'meta']):
                tag.decompose()
            return soup.get_text(" ")
        except:
            return re.sub(r'<[^>]+>', ' ', text)
    
    def remove_urls_emails(self, text: str) -> str:
        """Loại bỏ URLs và emails"""
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """Loại bỏ ký tự đặc biệt, giữ chữ tiếng Việt và số"""
        # Pattern giữ lại chữ cái tiếng Việt, số và khoảng trắng
        pattern = r'[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ_\s]'
        return re.sub(pattern, ' ', text)
    
    def normalize_repeated_characters(self, text: str) -> str:
        """Chuẩn hóa các ký tự lặp (vd: đẹpppp -> đẹp)"""
        return re.sub(r'(.)\1{2,}', r'\1', text)
    
    def remove_numbers_phone_id(self, text: str) -> str:
        """Loại bỏ số điện thoại, mã sản phẩm dài"""
        # Số điện thoại
        text = re.sub(r'0\d{9,10}', ' ', text)
        # Mã số dài (>6 chữ số)
        text = re.sub(r'\d{7,}', ' ', text)
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Chuẩn hóa khoảng trắng"""
        text = re.sub(r'[\n\t\r\v]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def word_segment(self, text: str) -> str:
        """Tách từ tiếng Việt"""
        if self._word_segment:
            try:
                return self._word_segment(text, format="text")
            except:
                return text
        return text
    
    def remove_stopwords_fn(self, text: str) -> str:
        """Loại bỏ stopwords"""
        words = text.split()
        filtered_words = [w for w in words if w.lower().replace('_', ' ') not in self.stopwords 
                         and w.lower() not in self.stopwords]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str) -> str:
        """
        Pipeline xử lý văn bản hoàn chỉnh
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản đã được xử lý
        """
        if not isinstance(text, str) or len(text) == 0:
            return ""
        
        # 1. Chuẩn hóa Unicode
        text = self.normalize_unicode(text)
        
        # 2. Chuyển thành chữ thường
        text = text.lower()
        
        # 3. Loại bỏ HTML tags
        text = self.remove_html_tags(text)
        
        # 4. Loại bỏ URLs và emails
        text = self.remove_urls_emails(text)
        
        # 5. Loại bỏ số điện thoại, mã số dài
        text = self.remove_numbers_phone_id(text)
        
        # 6. Chuẩn hóa ký tự lặp
        text = self.normalize_repeated_characters(text)
        
        # 7. Loại bỏ ký tự đặc biệt
        text = self.remove_special_characters(text)
        
        # 8. Chuẩn hóa khoảng trắng
        text = self.normalize_whitespace(text)
        
        # 9. Tách từ tiếng Việt (nếu enable)
        if self.use_word_segmentation:
            text = self.word_segment(text)
        
        # 10. Loại bỏ stopwords (nếu enable)
        if self.remove_stopwords:
            text = self.remove_stopwords_fn(text)
        
        # 11. Chuẩn hóa khoảng trắng lần cuối
        text = self.normalize_whitespace(text)
        
        return text
    
    def preprocess_batch(self, texts: List[str], show_progress: bool = True) -> List[str]:
        """
        Xử lý batch văn bản
        
        Args:
            texts: Danh sách văn bản
            show_progress: Hiển thị progress bar
            
        Returns:
            Danh sách văn bản đã xử lý
        """
        if show_progress:
            try:
                from tqdm import tqdm
                return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
            except ImportError:
                pass
        return [self.preprocess(text) for text in texts]


def combine_text_features(
    df: pd.DataFrame,
    columns: List[str] = ['product_name', 'description', 'brand'],
    separator: str = ' '
) -> pd.Series:
    """
    Kết hợp nhiều cột text thành một
    
    Args:
        df: DataFrame chứa dữ liệu
        columns: Danh sách các cột cần kết hợp
        separator: Ký tự phân cách
        
    Returns:
        Series chứa text đã kết hợp
    """
    combined = df[columns[0]].fillna('')
    for col in columns[1:]:
        combined = combined + separator + df[col].fillna('')
    return combined


def create_preprocessor(
    use_word_segmentation: bool = True,
    remove_stopwords: bool = True
) -> VietnameseTextPreprocessor:
    """
    Factory function để tạo preprocessor
    """
    return VietnameseTextPreprocessor(
        use_word_segmentation=use_word_segmentation,
        remove_stopwords=remove_stopwords
    )


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = create_preprocessor(use_word_segmentation=False, remove_stopwords=True)
    
    sample_text = """
    Set áo váy hồng tay hoa ngado - Set áo váy dài, set áo váy kiểu nữ.
    Tên sản phẩm: Set áo váy hồng tay hoa. Thông số: Size S M L XL.
    Chất liệu: Gân Ý. Giá sản phẩm trên Tiki đã bao gồm thuế theo luật hiện hành.
    """
    
    processed = preprocessor.preprocess(sample_text)
    print("Original:", sample_text[:100], "...")
    print("Processed:", processed[:100], "...")
