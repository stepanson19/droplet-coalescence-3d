# Слипание пары капель воды в невесомости — 3D

Отдельное учебное приложение на `Python + Streamlit + Plotly`, моделирующее слипание двух одинаковых капель воды в условиях невесомости с интерактивной 3D-визуализацией.

В проекте есть:

- математическая модель;
- интерфейс;
- вычислительный эксперимент.

## Что моделируется

Используется reduced-order модель из двух стадий:

1. рост жидкого мостика между двумя каплями;
2. релаксация объединенной капли к сфере с затухающей модой `l = 2`.

3D-геометрия строится как тело вращения вокруг оси слияния.

## Файлы

- `coalescence_core_3d.py` — математическая модель и 3D-геометрия;
- `web_app.py` — web-интерфейс;
- `self_test.py` — headless-проверка и генерация артефактов;
- `test_core_3d.py` — unit/smoke tests;
- `requirements.txt` — зависимости;
- `run_web.command` — запуск на macOS двойным кликом.

## Запуск

```bash
cd "/path/to/Василиса про"
python3 -m venv droplet_coalescence_3d/.venv
source droplet_coalescence_3d/.venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r droplet_coalescence_3d/requirements.txt
streamlit run droplet_coalescence_3d/web_app.py
```

После запуска откройте адрес вида `http://localhost:8501`.

## Публичный деплой для клиента

Самый прямой вариант для этого проекта: `Streamlit Community Cloud`.
Приложение уже собрано в подходящей структуре:

- входной файл: `droplet_coalescence_3d/web_app.py`;
- зависимости: `droplet_coalescence_3d/requirements.txt`.

Что нужно сделать:

1. Загрузить всю папку проекта в репозиторий GitHub.
2. Открыть [Streamlit Community Cloud](https://share.streamlit.io/).
3. Нажать `New app` и выбрать:
   - repository: ваш GitHub-репозиторий;
   - branch: обычно `main`;
   - main file path: `droplet_coalescence_3d/web_app.py`.
4. Запустить деплой и получить публичную ссылку вида `https://...streamlit.app`.

Локальная ссылка `http://localhost:8501` клиенту не подходит, потому что она видна только на вашем компьютере.
Для совпадения с облаком локально тоже лучше запускать `streamlit run droplet_coalescence_3d/web_app.py` из корня репозитория.

## Проверка без интерфейса

```bash
cd droplet_coalescence_3d
python3 -m pytest test_core_3d.py -q
python3 self_test.py
```

Будут созданы:

- `self_test_3d_surface.png`
- `self_test_experiment.png`
- `self_test_radius_sweep.csv`

## Ограничения

Это не прямой CFD/DNS-расчет системы Навье–Стокса, а физически осмысленная учебная reduced-order модель, подходящая для демонстрации, лабораторной или курсовой работы.
