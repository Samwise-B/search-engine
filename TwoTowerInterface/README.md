# Run the app directly
1. Open a terminal frontend dir and run: npm install; npm run dev
2. Open another terminal in backend dir and run: uvicorn app:app --reload  

# Run the app via Docker
1. Open a terminal in the root folder (TwoTowerInterface) and run: docker-compose up --build
2. After you've added the TwoTower Model to ./backend folder and linked it to fastapi run: docker push alberto1alberto/mlx:two_tower_model_with_interface
