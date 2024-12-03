from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import StringIO
import pandas as pd
import numpy as np
import joblib

# Загружаем обученную модель
model_8 = joblib.load("model_8.pkl")

# Определяем список признаков модели model_8 (предопределенные признаки, использованные при обучении)
model_8_features = ['engine_1047', 'engine_1061', 'engine_1086', 'engine_1120', 'engine_1150', 'engine_1172',
                'engine_1186', 'engine_1193', 'engine_1194', 'engine_1196', 'engine_1197', 'engine_1198',
                'engine_1199', 'engine_1248', 'engine_1298', 'engine_1299', 'engine_1339', 'engine_1341',
                'engine_1343', 'engine_1364', 'engine_1368', 'engine_1373', 'engine_1388', 'engine_1390',
                'engine_1396', 'engine_1399', 'engine_1405', 'engine_1422', 'engine_1451', 'engine_1461',
                'engine_1462', 'engine_1489', 'engine_1493', 'engine_1495', 'engine_1496', 'engine_1497',
                'engine_1498', 'engine_1499', 'engine_1527', 'engine_1582', 'engine_1586', 'engine_1590',
                'engine_1591', 'engine_1595', 'engine_1596', 'engine_1597', 'engine_1598', 'engine_1599',
                'engine_1781', 'engine_1794', 'engine_1795', 'engine_1796', 'engine_1797', 'engine_1798',
                'engine_1799', 'engine_1896', 'engine_1948', 'engine_1950', 'engine_1956', 'engine_1968',
                'engine_1969', 'engine_1984', 'engine_1991', 'engine_1994', 'engine_1995', 'engine_1997',
                'engine_1998', 'engine_1999', 'engine_2092', 'engine_2143', 'engine_2148', 'engine_2179',
                'engine_2197', 'engine_2198', 'engine_2199', 'engine_2200', 'engine_2231', 'engine_2354',
                'engine_2359', 'engine_2362', 'engine_2393', 'engine_2446', 'engine_2477', 'engine_2487',
                'engine_2489', 'engine_2494', 'engine_2496', 'engine_2497', 'engine_2498', 'engine_2499',
                'engine_2523', 'engine_2596', 'engine_2609', 'engine_2650', 'engine_2694', 'engine_2696',
                'engine_2755', 'engine_2776', 'engine_2835', 'engine_2953', 'engine_2956', 'engine_2967',
                'engine_2982', 'engine_2987', 'engine_2993', 'engine_2997', 'engine_2999', 'engine_3198',
                'engine_3498', 'engine_3604', 'engine_793', 'engine_796', 'engine_799', 'engine_814', 'engine_909',
                'engine_936', 'engine_993', 'engine_995', 'engine_998', 'engine_999', 'fuel_Diesel', 'fuel_LPG',
                'fuel_Petrol', 'km_driven', 'max_power', 'max_torque_rpm', 'mileage', 'name_Ashok', 'name_Audi',
                'name_BMW', 'name_Chevrolet', 'name_Daewoo', 'name_Datsun', 'name_Fiat', 'name_Force', 'name_Ford',
                'name_Honda', 'name_Hyundai', 'name_Isuzu', 'name_Jaguar', 'name_Jeep', 'name_Kia', 'name_Land',
                'name_Lexus', 'name_MG', 'name_Mahindra', 'name_Maruti', 'name_Mercedes-Benz', 'name_Mitsubishi',
                'name_Nissan', 'name_Opel', 'name_Peugeot', 'name_Renault', 'name_Skoda', 'name_Tata', 'name_Toyota',
                'name_Volkswagen', 'name_Volvo', 'owner_Fourth & Above Owner', 'owner_Second Owner',
                'owner_Test Drive Car', 'owner_Third Owner', 'seats_10', 'seats_14', 'seats_4', 'seats_5',
                'seats_6', 'seats_7', 'seats_8', 'seats_9', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
                'torque', 'transmission_Manual', 'year']  # Полный список 175 признаков

# Инициализируем FastAPI приложение
app = FastAPI()

# Определяем Pydantic модель
class Car(BaseModel):
    name: str
    year: int
    km_driven: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: int
    max_power: float
    torque: float
    seats: int
    max_torque_rpm: float

@app.post("/predict_item")
def predict_item(car: Car):
    # Шаг 1: Конвертируем JSON в DataFrame
    df_test = pd.DataFrame([car.dict()])  # Преобразуем данные из модели Pydantic в DataFrame

    # Шаг 2: Инженерия признаков (должна быть консистентной с обучением)
    X_test_cat_mod = df_test.copy()
    X_test_cat_mod['year'] = df_test['year'].apply(lambda x: x ** 2)  # Признак 1
    X_test_cat_mod['max_torque_rpm'] = df_test['max_torque_rpm'].apply(
        lambda x: np.log(x) if x != 0 else 0)  # Признак 2

    # Кодирование категориальных переменных
    X_test_cat_mod_enc = pd.get_dummies(X_test_cat_mod,
                                        columns=['name', 'engine', 'fuel', 'seller_type', 'transmission', 'owner',
                                                 'seats'],
                                        drop_first=True,
                                        dtype=float)

    # Убедимся, что все столбцы совпадают с набором данных для обучения
    all_columns = list(set(X_test_cat_mod_enc.columns).union(set(model_8_features)))
    X_test_cat_mod_enc = X_test_cat_mod_enc.reindex(sorted(all_columns), fill_value=0, axis=1)

    # Шаг 3: Предсказание с использованием модели
    prediction = model_8.predict(X_test_cat_mod_enc)

    # Шаг 4: Возвращаем предсказание
    return {"predicted_price": round(float(prediction[0]), 1)}

@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    # Шаг 1: Читаем и разбираем загруженный файл
    contents = await file.read()
    lines = contents.decode("utf-8").splitlines()
    df_test = pd.read_csv(StringIO("\n".join(lines)))

    # Шаг 2: Feature engineering (преобразовали уже имеющиеся признаки)
    X_test_cat_mod = df_test.copy()
    X_test_cat_mod['year'] = df_test['year'].apply(lambda x: x ** 2)  # Признак 1
    X_test_cat_mod['max_torque_rpm'] = df_test['max_torque_rpm'].apply(
        lambda x: np.log(x) if x != 0 else 0)  # Признак 2

    # Кодирование категориальных переменных
    X_test_cat_mod_enc = pd.get_dummies(X_test_cat_mod,
                                        columns=['name', 'engine', 'fuel', 'seller_type', 'transmission', 'owner',
                                                 'seats'],
                                        drop_first=True,
                                        dtype=float)

    # Выравниваем с тренировочными колонками
    all_columns = list(set(X_test_cat_mod_enc.columns).union(set(model_8_features)))
    X_test_cat_mod_enc = X_test_cat_mod_enc.reindex(sorted(all_columns), fill_value=0, axis=1)

    # Шаг 3: Предсказание с использованием модели
    df_test['selling_price'] = model_8.predict(X_test_cat_mod_enc)

    # Шаг 4: Преобразуем DataFrame в CSV для ответа
    output = StringIO()
    df_test.to_csv(output, index=False)
    output.seek(0)

    # Возвращаем файл в виде скачиваемого ответа
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment;filename=predictions.csv"})


