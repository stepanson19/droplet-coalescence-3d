# Моделирование слипания капель воды

Репозиторий содержит две учебные версии модели слипания капель воды в невесомости:

- `droplet_coalescence/` — исходная 2D-версия;
- `droplet_coalescence_3d/` — отдельная 3D-версия с интерфейсом и вычислительным экспериментом.

## 3D-версия

Папка `droplet_coalescence_3d/` содержит приложение на `Python + Streamlit + Plotly`, в котором есть:

- математическая модель двухстадийного слипания;
- интерактивная 3D-визуализация;
- вычислительный эксперимент с выгрузкой данных.

Локальный запуск:

```bash
cd "/path/to/Василиса про"
python3 -m venv droplet_coalescence_3d/.venv
source droplet_coalescence_3d/.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r droplet_coalescence_3d/requirements.txt
streamlit run droplet_coalescence_3d/web_app.py
```

Подробнее: `droplet_coalescence_3d/README.md`.

## Публичный доступ для клиента

Для постоянной внешней ссылки этот репозиторий подготовлен под `Streamlit Community Cloud`.

Нужные параметры деплоя:

- repository: этот GitHub-репозиторий;
- branch: `main`;
- main file path: `droplet_coalescence_3d/web_app.py`.
