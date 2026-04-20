"""Hybrid Prompt Generation — CASDA src/preprocessing/prompt_generator.py 포팅.

결함 기하 지표(linearity, solidity, aspect_ratio)와 배경 유형을
ControlNet 텍스트 조건으로 변환한다.

CASDA 원본 대비 변경:
- class_id 제거 (AROMA는 defect_subtype 문자열로 결함 유형 식별)
- "steel" 등 도메인 고유 표현 제거 → 도메인 중립 프롬프트
- generate_technical_prompt()를 AROMA 기본 스타일로 지정
"""
from typing import Dict, List


class PromptGenerator:
    DEFECT_DESCRIPTIONS = {
        'linear_scratch': {
            'base': 'a linear scratch defect',
            'detailed': 'a high-linearity elongated scratch',
            'characteristics': ['linear', 'scratch-like', 'elongated'],
        },
        'elongated': {
            'base': 'an elongated defect',
            'detailed': 'a moderately elongated defect region',
            'characteristics': ['elongated', 'stretched'],
        },
        'compact_blob': {
            'base': 'a compact blob defect',
            'detailed': 'a solid compact defect spot',
            'characteristics': ['compact', 'blob-like', 'solid'],
        },
        'irregular': {
            'base': 'an irregular defect',
            'detailed': 'an irregular defect with complex boundaries',
            'characteristics': ['irregular', 'complex-shaped'],
        },
        'general': {
            'base': 'a surface defect',
            'detailed': 'a general surface defect',
            'characteristics': ['defect'],
        },
    }

    BACKGROUND_DESCRIPTIONS = {
        'smooth': {
            'surface': 'smooth surface',
            'texture': 'uniform texture',
            'pattern': 'no visible pattern',
        },
        'textured': {
            'surface': 'textured surface',
            'texture': 'grainy texture',
            'pattern': 'subtle surface texture',
        },
        'directional': {
            'surface': 'directional striped surface',
            'texture': 'directional texture',
            'pattern': 'directional line pattern',
        },
        'vertical_stripe': {
            'surface': 'vertical striped surface',
            'texture': 'directional texture',
            'pattern': 'vertical line pattern',
        },
        'horizontal_stripe': {
            'surface': 'horizontal striped surface',
            'texture': 'directional texture',
            'pattern': 'horizontal line pattern',
        },
        'periodic': {
            'surface': 'periodic patterned surface',
            'texture': 'repeating texture',
            'pattern': 'periodic surface pattern',
        },
        'complex_pattern': {
            'surface': 'complex patterned surface',
            'texture': 'multi-directional texture',
            'pattern': 'complex surface pattern',
        },
    }

    SURFACE_QUALITY = {
        'high': ['pristine', 'well-maintained', 'clean'],
        'medium': ['standard', 'typical', 'normal'],
        'low': ['worn', 'weathered', 'aged'],
    }

    def __init__(self, style: str = 'technical'):
        self.style = style

    def _surface_quality(self, stability_score: float) -> str:
        import random
        if stability_score >= 0.8:
            return random.choice(self.SURFACE_QUALITY['high'])
        elif stability_score >= 0.5:
            return random.choice(self.SURFACE_QUALITY['medium'])
        return random.choice(self.SURFACE_QUALITY['low'])

    def generate_simple_prompt(self, defect_subtype: str, background_type: str) -> str:
        defect_desc = self.DEFECT_DESCRIPTIONS.get(
            defect_subtype, self.DEFECT_DESCRIPTIONS['general']
        )['base']
        bg_desc = self.BACKGROUND_DESCRIPTIONS.get(
            background_type, self.BACKGROUND_DESCRIPTIONS['smooth']
        )['surface']
        return f"{defect_desc} on {bg_desc}"

    def generate_detailed_prompt(
        self,
        defect_subtype: str,
        background_type: str,
        stability_score: float,
        defect_metrics: Dict,
    ) -> str:
        defect_desc = self.DEFECT_DESCRIPTIONS.get(
            defect_subtype, self.DEFECT_DESCRIPTIONS['general']
        )['detailed']
        bg_info = self.BACKGROUND_DESCRIPTIONS.get(
            background_type, self.BACKGROUND_DESCRIPTIONS['smooth']
        )
        surface_quality = self._surface_quality(stability_score)
        parts = [defect_desc, "on", bg_info['surface'], "with", bg_info['texture']]
        if stability_score >= 0.6:
            parts.append(f"({surface_quality} condition)")
        return " ".join(parts)

    def generate_technical_prompt(
        self,
        defect_subtype: str,
        background_type: str,
        stability_score: float,
        defect_metrics: Dict,
        suitability_score: float,
    ) -> str:
        """linearity / solidity / aspect_ratio → 자연어 특성 변환 (CASDA 핵심 로직)."""
        defect_info = self.DEFECT_DESCRIPTIONS.get(
            defect_subtype, self.DEFECT_DESCRIPTIONS['general']
        )
        bg_info = self.BACKGROUND_DESCRIPTIONS.get(
            background_type, self.BACKGROUND_DESCRIPTIONS['smooth']
        )

        linearity = defect_metrics.get('linearity', 0.0)
        solidity = defect_metrics.get('solidity', 0.0)
        aspect_ratio = defect_metrics.get('aspect_ratio', 1.0)

        characteristics = []
        if linearity > 0.8:
            characteristics.append("highly linear")
        elif linearity > 0.6:
            characteristics.append("moderately linear")
        if solidity > 0.9:
            characteristics.append("solid")
        elif solidity < 0.7:
            characteristics.append("irregular shape")
        if aspect_ratio > 5.0:
            characteristics.append("very elongated")
        elif aspect_ratio > 3.0:
            characteristics.append("elongated")

        char_str = ", ".join(characteristics) if characteristics else defect_info['characteristics'][0]

        return (
            f"Industrial defect: {char_str} {defect_subtype.replace('_', ' ')} "
            f"on {bg_info['surface']}, "
            f"{bg_info['pattern']}, "
            f"background stability {stability_score:.2f}, "
            f"match quality {suitability_score:.2f}"
        )

    def generate_prompt(
        self,
        defect_subtype: str,
        background_type: str,
        stability_score: float = 0.5,
        defect_metrics: Dict = None,
        suitability_score: float = 0.5,
    ) -> str:
        if defect_metrics is None:
            defect_metrics = {}
        if self.style == 'simple':
            return self.generate_simple_prompt(defect_subtype, background_type)
        elif self.style == 'detailed':
            return self.generate_detailed_prompt(
                defect_subtype, background_type, stability_score, defect_metrics
            )
        return self.generate_technical_prompt(
            defect_subtype, background_type, stability_score, defect_metrics, suitability_score
        )

    def generate_negative_prompt(self) -> str:
        return (
            "blurry, low quality, artifacts, noise, distorted, "
            "warped, unrealistic, oversaturated, color bleeding, shadow artifacts"
        )

    def batch_generate_prompts(self, roi_metadata: List[Dict]) -> List[Dict]:
        for roi_data in roi_metadata:
            roi_data['prompt'] = self.generate_prompt(
                defect_subtype=roi_data.get('subtype', 'general'),
                background_type=roi_data.get('background_type', 'smooth'),
                stability_score=roi_data.get('stability_score', 0.5),
                defect_metrics=roi_data,
                suitability_score=roi_data.get('suitability_score', 0.5),
            )
            roi_data['negative_prompt'] = self.generate_negative_prompt()
        return roi_metadata
