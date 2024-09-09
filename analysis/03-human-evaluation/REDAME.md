# Evaluation System

This is a Flask-based web application for evaluating various dishes. The application allows users to evaluate dishes, automatically records the time taken for the evaluation, and saves the evaluations in a SQLite database. The evaluations can be exported into JSON files, organized by user ID and dish name.

## Features

- **User Evaluation**: Users can evaluate dishes, with their evaluations being stored in a SQLite database.
- **Time Tracking**: The system automatically tracks the time taken for each evaluation session.
- **Data Export**: Evaluations can be exported to JSON files, organized by user ID and dish name, with each user's data saved in a separate directory.
- **Data Deletion**: Users can delete all evaluation records from the database.

## Requirements

- Python 3.x
- Flask
- Flask-SQLAlchemy

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/evaluation-system.git
   cd evaluation-system
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

### 1. Evaluate a Dish

- Navigate to the homepage and select a cultural group.
- The application will assign a unique user ID for your session.
- Evaluate the dishes presented to you by selecting options and providing feedback.

### 2. Track Evaluation Time

- The system will automatically start tracking the time when you begin evaluating.
- After completing the evaluation of all dishes, the total time spent will be calculated and stored in the database.

### 3. Export Evaluations

To export the evaluations as JSON files, visit the following URL:

```url
http://127.0.0.1:5000/export-evaluations
```

- JSON files will be created for each user in separate directories named after their user ID. Each file will contain evaluations for individual dishes.

### 4. Directory Structure

The exported data will be saved in a structure like:

```bash
/user_<user_id>/
    ├── dish_name_1_evaluations_output.json
    ├── dish_name_2_evaluations_output.json
    └── ...
```

### 5. Delete All Evaluations

To delete all evaluation records from the database, you can use the following URL:

```url
http://127.0.0.1:5000/clear-evaluations
```

- This will remove all evaluation records permanently from the database.

## Routes

- **`/cultural-group`**: Select a cultural group and start the evaluation.
- **`/evaluate-dish/<group>`**: Evaluate the dishes from the selected cultural group.
- **`/evaluation-time/<group>`**: Calculate and store the total evaluation time.
- **`/export-evaluations`**: Export all evaluations into JSON files, organized by user ID and dish name.
- **`/clear-evaluations`**: Delete all evaluation records from the database.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
