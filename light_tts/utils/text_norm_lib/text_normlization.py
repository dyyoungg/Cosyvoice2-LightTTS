# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import List

from .char_convert import tranditional_to_simplified
from .chronology import RE_DATE
from .chronology import RE_DATE2
from .chronology import RE_TIME
from .chronology import RE_TIME_RANGE
from .chronology import replace_date
from .chronology import replace_date2
from .chronology import replace_time
from .constants import F2H_ASCII_LETTERS
from .constants import F2H_DIGITS
from .constants import F2H_SPACE
from .num import RE_DECIMAL_NUM
from .num import RE_DEFAULT_NUM
from .num import RE_FRAC
from .num import RE_INTEGER
from .num import RE_NUMBER
from .num import RE_PERCENTAGE
from .num import RE_POSITIVE_QUANTIFIERS
from .num import RE_RANGE
from .num import replace_default_num
from .num import replace_frac
from .num import replace_negative_num
from .num import replace_number
from .num import replace_percentage
from .num import replace_positive_quantifier
from .num import replace_range
from .phonecode import RE_MOBILE_PHONE
from .phonecode import RE_NATIONAL_UNIFORM_NUMBER
from .phonecode import RE_TELEPHONE
from .phonecode import replace_mobile
from .phonecode import replace_phone
from .quantifier import RE_TEMPERATURE
from .quantifier import replace_measure
from .quantifier import replace_temperature


def replace_money(text: str) -> str:
    # 美元
    text = re.sub(r'\$\s*(\d+(?:\.\d+)?)', r'\1美元', text)
    # 人民币
    text = re.sub(r'¥\s*(\d+(?:\.\d+)?)', r'\1元', text)
    # 欧元
    text = re.sub(r'€\s*(\d+(?:\.\d+)?)', r'\1欧元', text)
    # 英镑
    text = re.sub(r'£\s*(\d+(?:\.\d+)?)', r'\1英镑', text)
    # 日元
    text = re.sub(r'¥\s*(\d+(?:\.\d+)?)', r'\1日元', text)
    # 韩元
    text = re.sub(r'₩\s*(\d+(?:\.\d+)?)', r'\1韩元', text)
    # 卢布
    text = re.sub(r'₽\s*(\d+(?:\.\d+)?)', r'\1卢布', text)
    # 印度卢比
    text = re.sub(r'₹\s*(\d+(?:\.\d+)?)', r'\1印度卢比', text)
    # 澳元
    text = re.sub(r'A\$\s*(\d+(?:\.\d+)?)', r'\1澳元', text)
    # 加元
    text = re.sub(r'C\$\s*(\d+(?:\.\d+)?)', r'\1加元', text)
    return text

def replace_special_format(match):
    return f"{match.group(1)}{match.group(2)}"  # 保持原格式
    

class TextNormalizer():
    def __init__(self):
        self.SENTENCE_SPLITOR = re.compile(r'([：、，；。？！,;?!][”’]?)')
        self.RE_SPECIAL_FORMAT = re.compile(r'([A-Za-z]+)-(\d+)') # GPT-5, Windows-11，iphone15

    def _split(self, text: str, lang="zh") -> List[str]:
        """Split long text into sentences with sentence-splitting punctuations.
        Args:
            text (str): The input text.
        Returns:
            List[str]: Sentences.
        """
        # Only for pure Chinese here
        if lang == "zh":
            text = text.replace(" ", "")
            # 过滤掉特殊字符
            text = re.sub(r'[——《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
        text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
        return sentences

    def _post_replace(self, sentence: str) -> str:
        sentence = sentence.replace('/', '每')
        # sentence = sentence.replace('~', '至')
        # sentence = sentence.replace('～', '至')
        sentence = sentence.replace('①', '一')
        sentence = sentence.replace('②', '二')
        sentence = sentence.replace('③', '三')
        sentence = sentence.replace('④', '四')
        sentence = sentence.replace('⑤', '五')
        sentence = sentence.replace('⑥', '六')
        sentence = sentence.replace('⑦', '七')
        sentence = sentence.replace('⑧', '八')
        sentence = sentence.replace('⑨', '九')
        sentence = sentence.replace('⑩', '十')
        sentence = sentence.replace('α', '阿尔法')
        sentence = sentence.replace('β', '贝塔')
        sentence = sentence.replace('γ', '伽玛').replace('Γ', '伽玛')
        sentence = sentence.replace('δ', '德尔塔').replace('Δ', '德尔塔')
        sentence = sentence.replace('ε', '艾普西龙')
        sentence = sentence.replace('ζ', '捷塔')
        sentence = sentence.replace('η', '依塔')
        sentence = sentence.replace('θ', '西塔').replace('Θ', '西塔')
        sentence = sentence.replace('ι', '艾欧塔')
        sentence = sentence.replace('κ', '喀帕')
        sentence = sentence.replace('λ', '拉姆达').replace('Λ', '拉姆达')
        sentence = sentence.replace('μ', '缪')
        sentence = sentence.replace('ν', '拗')
        sentence = sentence.replace('ξ', '克西').replace('Ξ', '克西')
        sentence = sentence.replace('ο', '欧米克伦')
        sentence = sentence.replace('π', '派').replace('Π', '派')
        sentence = sentence.replace('ρ', '肉')
        sentence = sentence.replace('ς', '西格玛').replace('Σ', '西格玛').replace(
            'σ', '西格玛')
        sentence = sentence.replace('τ', '套')
        sentence = sentence.replace('υ', '宇普西龙')
        sentence = sentence.replace('φ', '服艾').replace('Φ', '服艾')
        sentence = sentence.replace('χ', '器')
        sentence = sentence.replace('ψ', '普赛').replace('Ψ', '普赛')
        sentence = sentence.replace('ω', '欧米伽').replace('Ω', '欧米伽')
        #算术符
        sentence = sentence.replace('+', '加')
        sentence = sentence.replace('-', '减')
        sentence = sentence.replace('×', '乘')
        sentence = sentence.replace('÷', '除以')
        sentence = sentence.replace('=', '等于')
        sentence = sentence.replace('<', '小于')
        sentence = sentence.replace('>', '大于')
        sentence = sentence.replace('≥', '大于等于')
        sentence = sentence.replace('≤', '小于等于')
        # re filter special characters, have one more character "-" than line 68
        sentence = re.sub(r'[-——《》【】<=>{}()（）#&@“”^_|…\\]', '', sentence)
        return sentence


    def normalize_sentence(self, sentence: str) -> str:
        # basic character conversions
        sentence = tranditional_to_simplified(sentence)
        sentence = sentence.translate(F2H_ASCII_LETTERS).translate(
            F2H_DIGITS).translate(F2H_SPACE)
        sentence = re.sub(r'(?<=\d)[~～](?=\d)', '到', sentence)
        sentence = replace_money(sentence)
        # number related NSW verbalization
        sentence = self.RE_SPECIAL_FORMAT.sub(replace_special_format, sentence)
     
        sentence = RE_DATE.sub(replace_date, sentence)
        sentence = RE_DATE2.sub(replace_date2, sentence)
        # range first
        sentence = RE_TIME_RANGE.sub(replace_time, sentence)
        sentence = RE_TIME.sub(replace_time, sentence)
      
        sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
      
        sentence = replace_measure(sentence)
        # print(sentence)
        sentence = RE_FRAC.sub(replace_frac, sentence)
        # print(sentence)
    
        sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
        # print(sentence)
        sentence = RE_MOBILE_PHONE.sub(replace_mobile, sentence)
        # print(sentence)

        sentence = RE_TELEPHONE.sub(replace_phone, sentence)
        # print(sentence)
        sentence = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, sentence)
        # print(sentence)
        sentence = RE_RANGE.sub(replace_range, sentence)
       
        sentence = RE_INTEGER.sub(replace_negative_num, sentence)
   
        sentence = RE_DECIMAL_NUM.sub(replace_number, sentence)
  
        sentence = RE_POSITIVE_QUANTIFIERS.sub(replace_positive_quantifier, sentence)
      
        sentence = RE_DEFAULT_NUM.sub(replace_default_num, sentence)
     
        sentence = RE_NUMBER.sub(replace_number, sentence)
        sentence = self._post_replace(sentence)

        return sentence

    def normalize(self, text: str) -> List[str]:
        # sentences = self._split(text) # 暂时不支持split
        sentences = self.normalize_sentence(text)
        return sentences


if __name__ == "__main__":
    normalizer = TextNormalizer()
    text = "2024-03-19，OpenAI 发布了 GPT-5，售价为 $900，支持 iOS 和 Android 系统。售价2024元，只有苹果的1/3"
    test_cases = [
        # 日期和时间
        "防护服的充能条上显示的呀，上面写着“当前防护等级15/80”，所以还差7个钠才能充满啊。",
        "2024-03-19 15:30",
        "2023年12月25日",
        "08:00-17:00",
        # 手机号码
        "我的电话是15909234343",
        # 数字和货币
        "$999.99",
        "¥ 2024",
        "100%",
        "3.14",
        "1/2",
        "2024元",
        "3.5cm",
        "90232m",
        # 特殊格式
        "GPT-5发布",
        "iPhone-15 Pro",
        "Windows-11",
        
        # 符号和单位
        "25℃",
        "36.5°C",
        "1~10",
        "1～10",
        
        # 混合文本
        "2024-03-19，OpenAI 发布了 GPT-5，售价为$999.99，支持 iOS 和 Android 系统。售价2024元",
        "今天气温25℃，湿度50%，风速10m/cm2",
        "iPhone-15 Pro 售价 ¥9999，支持 5G 网络",
        "风速为10m/s，温度为25℃",  
        "密度为5kg/m³，压力为100Pa", 
        "加速度为9.8m/s²，功率为100W", 
        "iPhone-15 Pro 的厚度为7.85mm，重量为187g，电池容量为3274mAh", 
        "你知道3÷5-2=多少吗, 5<=2",
        "电影中梁朝伟扮演的陈永仁的编号27149",
        "你好啊~我真喜欢你",
        # 边界情况
        "",
        "Hello, World!",
        "123abc456",
        "αβγδεζηθικλμνξοπρστυφχψω",
        "《》【】（）{}"
    ]

    import time
    normalizer = TextNormalizer()
    for text in test_cases:
        t1 = time.time()
        normalized_text = normalizer.normalize_sentence(text)
        print(f"Input: {text} Output: {normalized_text}", time.time() - t1)
        print("-" * 50)