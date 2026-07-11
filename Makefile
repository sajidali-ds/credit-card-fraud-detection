.PHONY: install train evaluate app mlflow-ui test docker-build docker-up docker-down lint clean

install:
	pip install -r requirements.txt

train:
	python main.py

evaluate:
	python -m src.evaluate

app:
	streamlit run app.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	pytest tests/ -v

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
