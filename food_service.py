from .firestore_client import db

# Hàm lấy toàn bộ dữ liệu món ăn từ Firestore

def get_all_foods():
    """
    Lấy toàn bộ dữ liệu món ăn từ Firestore.
    Trả về: List[dict]
    """
    foods_ref = db.collection('food')
    docs = foods_ref.stream()
    food_list = []
    for doc in docs:
        food = doc.to_dict()
        food['id'] = doc.id
        food_list.append(food)
    return food_list

# (Tùy chọn) Cache dữ liệu để tăng tốc
import time
_food_cache = None
_cache_time = 0
_CACHE_TTL = 300  # giây

def get_all_foods_cached():
    global _food_cache, _cache_time
    now = time.time()
    if _food_cache is None or now - _cache_time > _CACHE_TTL:
        _food_cache = get_all_foods()
        _cache_time = now
    return _food_cache
