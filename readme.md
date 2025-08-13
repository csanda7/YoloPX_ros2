# YOLOPX TensorRT ROS2 Node

Ez a projekt egy ROS2 node, amely TensorRT segítségével végez valós idejű képfeldolgozást YOLOPX modellel. A node képes **sensor\_msgs/Image** és **sensor\_msgs/CompressedImage** típusú topicokat feldolgozni, és automatikusan a *CompressedImage* verziót részesíti előnyben, ha elérhető.
A topicra feliratkozva a vezethető útfelületet illetve az útra való felfestéseket szegmentálja.

## Követelmények

- **ROS 2 Humble** 
- **TensorRT** telepítve NVIDIA forrásból (pip/apt/wheel)
- NVIDIA GPU és CUDA driver kompatibilis a TensorRT-vel
- **TensorRT verzió: 10.9.0.34**

Python csomagok telepítése:

```bash
pip install -r requirements.txt
```


## Konfiguráció

A `config.py` fájlban adhatod meg:

- `ENGINE` – TensorRT engine fájl elérési útja
- `TOPIC` – alap topic név (pl. `/camera/image_raw` vagy `/camera/image_raw/compressed`)
- `LANE_THRESH`, `DRIVE_THRESH` – maszk küszöbértékek
- `SHOW` – megjelenítés kapcsoló
- `RELIABLE` – QoS beállítás

## Futtatás

A node indítása:

```bash
ros2 launch yolopx seg_sub_trt
```


## Működés

1. Induláskor a node ellenőrzi, hogy elérhető-e a megadott `TOPIC` *CompressedImage* változata.
2. Ha igen, erre iratkozik fel, különben a sima Image változatra.
3. Ha később megjelenik a CompressedImage, automatikusan átvált rá.
4. A képkockákat előfeldolgozza, TensorRT-vel inferálja, majd maszkolt eredményt jelenít meg (ha engedélyezve van).


