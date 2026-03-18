import pandas as pd

# 1. Загружаем файл
df = pd.read_csv("коргау_clean.csv", sep=";", encoding="utf-8-sig")

# 2. Объединяем Дату и Время
# errors='coerce' превратит битые даты в NaT (чтобы скрипт не падал)
# dayfirst=True важен, так как у тебя формат ДД.ММ.ГГГГ
dt_series = pd.to_datetime(
    df['Дата'].astype(str) + ' ' + df['Время'].astype(str), 
    dayfirst=True, 
    errors='coerce'
)

# 3. Создаем новый столбец в формате 2013-02-20 06:00:00.000
df['Дата_Время'] = dt_series.dt.strftime('%Y-%m-%d %H:%M:%S.000')

# 4. Удаляем старые столбцы
df = df.drop(columns=['Дата', 'Время'])

# 5. Сохраняем результат
df.to_csv("коргау_final.csv", sep=";", encoding="utf-8-sig", index=False)

print("Готово! Столбцы Дата и Время объединены.")
print(df['Дата_Время'].head())