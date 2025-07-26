# config/settings.py
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FileSettings:
    CATS_FILE: str = './0.datafiles/org_catsdb.xlsx'
    ENEMY_FILE: str = './0.datafiles/nyanko_enemy_db.xlsx'

@dataclass
class ColumnSettings:
    NUMERIC_COLS_CATS: List[str] = field(default_factory=lambda: [
        'Own', 'No.', 'コスト', '再生産F', '速度', '射程', '発生F',
        '攻撃力', '頻度F', 'DPS', '体力', 'KB'
    ])
    NUMERIC_COLS_ENEMY: List[str] = field(default_factory=lambda: [
        '体力', 'KB', '速度', '攻撃力', 'DPS', '頻度F', '攻発F', '射程', 'お金'
    ])
    DISPLAY_COLS_CATS: List[str] = field(default_factory=lambda: [
        'Own', 'No.', 'ランク', 'キャラクター名', 'コスト', '再生産F',
        '速度', '範囲', '射程', '発生F', '攻撃力', '頻度F', 'DPS',
        '体力', 'KB', '特性'
    ])

@dataclass
class TraitSettings:
    COLOR_TRAITS: List[str] = field(default_factory=lambda: [
        '赤', '浮', '黒', 'メタル', '天使', 'エイリアン',
        'ゾンビ', '古代種', '悪魔', '白'
    ])
    BOOLEAN_TRAITS: Dict[str, str] = field(default_factory=lambda: {
        'めっぽう強い': 'めっぽう強い',
        '打たれ強い': '打たれ強い',
        '超打たれ強い': '超打たれ強い',
        '超ダメージ': '超ダメージ',
        '極ダメージ': '極ダメージ',
        'ターゲット限定': 'のみに攻撃',
        '魂攻撃': '魂攻撃',
        'メタルキラー': 'メタルキラー',
        '被ダメージ1': r'被ダメージ\s*1',
        '波動ストッパー': '波動ストッパー',
        '烈波カウンター': '烈波カウンター',
        '1回攻撃': '1回攻撃',
        'ゾンビキラー': 'ゾンビキラー',
        'バリアブレイク': 'バリアブレイク',
        '悪魔シールド貫通': '悪魔シールド貫通',
    })
    FLAG_TRAITS: List[str] = field(default_factory=lambda: [
        '攻撃力低下', '動きを止める', '動きを遅くする', 'ふっとばす',
        '呪い', '攻撃無効', '渾身の一撃', '攻撃力上昇', '生き残る',
        'クリティカル', '波動', '小波動', '烈波', '小烈波', '爆波',
    ])

@dataclass
class UISettings:
    RANK_OPTIONS: List[str] = field(default_factory=lambda: [
        '基本', 'EX', 'レア', '激レア', '超激レア', '伝説レア'
    ])
    RANGE_OPTIONS: List[str] = field(default_factory=lambda: ['単体', '範囲'])
    SPECIAL_EFFECTS: List[str] = field(default_factory=lambda: [
        'めっぽう強い', '打たれ強い', '超ダメージ', '攻撃力低下',
        '動きを止める', '動きを遅くする', 'ふっとばす', '呪い', '攻撃無効'
    ])
    SPECIAL_ABILITIES: List[str] = field(default_factory=lambda: [
        '波動', '小波動', '烈波', '小烈波', '爆波',
        'クリティカル', '渾身の一撃', 'ゾンビキラー', '悪魔シールド貫通',
        'バリアブレイク', '生き残る', '波動ストッパー'
    ])