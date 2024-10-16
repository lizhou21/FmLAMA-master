from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import json
import uuid
import os
from datetime import datetime, timezone

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///evaluations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 


db = SQLAlchemy(app)


class Evaluation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100)) 
    origin = db.Column(db.String(100))  
    group = db.Column(db.String(100))
    dish_name = db.Column(db.String(100))
    predicted_ingredient = db.Column(db.String(100))
    correct = db.Column(db.String(10))
    correct_reason = db.Column(db.String(200), nullable=True)
    evaluation_time = db.Column(db.String(20), nullable=True)  
    comment = db.Column(db.String(500), nullable=True)


with app.app_context():
    db.create_all()


with open('English_human_evaluation.json', encoding='utf-8') as f:
    dishes_data = json.load(f)


@app.route('/', methods=['GET', 'POST'])
def guidelines():
    if request.method == 'POST':
        confirm1 = request.form.get('confirm1')
        confirm2 = request.form.get('confirm2')

        if confirm1 and confirm2:
            return redirect(url_for('cultural_group'))
        else:
            error = "Please confirm both checkboxes to proceed."
            return render_template('guidelines.html', error=error)

    return render_template('guidelines.html')


@app.route('/cultural-group', methods=['GET', 'POST'])
def cultural_group():

    if 'user_id' not in session:
        origin = request.form.get('origin', 'unknown_origin').replace(" ", "_")
        session['origin'] = origin  
        user_id = f"{origin}_{uuid.uuid4().hex[:8]}"
        session['user_id'] = user_id

    user_id = session['user_id']  
    cultural_groups = list(dishes_data.keys())  
    
    if request.method == 'POST':
        selected_group = request.form.get('cultural_group')
        if selected_group:
            return redirect(url_for('evaluate_dish', group=selected_group))
        else:
            error = "Please select a cultural group."
            return render_template('cultural_group.html', groups=cultural_groups, error=error)

    return render_template('cultural_group.html', groups=cultural_groups)

@app.route('/evaluate-dish/<group>', methods=['GET', 'POST'])
def evaluate_dish(group):
    user_id = session['user_id']  
      
    dishes = dishes_data[group][:2]
    dish_index = int(request.args.get('dish_index', 0))  
    current_dish = dishes[dish_index]
    origin = current_dish["origin"]  # 


    if 'dish_start_time' not in session:
        session['dish_start_time'] = datetime.now(timezone.utc)  

    if request.method == 'POST':
     
        dish_end_time = datetime.now(timezone.utc) 
        dish_start_time = session.pop('dish_start_time', None)  
        
        if dish_start_time:
            dish_time_spent = dish_end_time - dish_start_time 


            comment = request.form.get('comment')


            for i, predicted_ingredient in enumerate(current_dish['predicted_ingredient'], start=1):
                ingredient_value = request.form.get(f'ingredient_{i}')
                correct_reason = request.form.get(f'correct_reason_{i}', None)

                evaluation = Evaluation(
                    user_id=user_id, 
                    origin=origin,  
                    group=group,
                    dish_name=current_dish['dish'],
                    predicted_ingredient=predicted_ingredient,
                    correct=ingredient_value,
                    correct_reason=correct_reason,
                    evaluation_time=str(dish_time_spent), 
                    comment=comment
                )
                db.session.add(evaluation)
            db.session.commit()


        if dish_index < len(dishes) - 1:
            session['dish_start_time'] = datetime.now(timezone.utc)  
            return redirect(url_for('evaluate_dish', group=group, dish_index=dish_index + 1))
        else:
            return redirect(url_for('evaluation_time', group=group))

    return render_template('evaluate_dish.html', dish=current_dish)


@app.route('/evaluation-time/<group>', methods=['GET', 'POST'])
def evaluation_time(group):
    user_id = session['user_id'] 
    origin = session.get('origin', 'unknown_origin') 

    return f"Thank you for your evaluation!"


@app.route('/export-evaluations', methods=['GET'])
def export_evaluations():

    user_ids = db.session.query(Evaluation.user_id).distinct().all()
    

    for i, user_id_tuple in enumerate(user_ids):
        user_id = user_id_tuple[0]
        user_evaluations = db.session.query(Evaluation).filter_by(user_id=user_id).all()
        user_origin = user_evaluations[0].origin if user_evaluations[0].origin else 'unknown_origin'
        user_dir = f'user_'+ user_id + "_" + user_origin 
        print(user_dir)


        if not os.path.exists(user_dir):
            os.makedirs(user_dir)


        dish_names = db.session.query(Evaluation.dish_name).filter_by(user_id=user_id).distinct().all()
        for dish_name_tuple in dish_names:
            dish_name = dish_name_tuple[0]
            evaluations = Evaluation.query.filter_by(user_id=user_id, dish_name=dish_name).all()
            

            evaluations_list = []
            for evaluation in evaluations:
                evaluations_list.append({
                    'origin': evaluation.origin,
                    'group': evaluation.group,
                    'dish_name': evaluation.dish_name,
                    'predicted_ingredient': evaluation.predicted_ingredient,
                    'correct': evaluation.correct,
                    'correct_reason': evaluation.correct_reason,
                    'evaluation_time': evaluation.evaluation_time,
                    "comment": evaluation.comment
                })


            sanitized_dish_name = dish_name.replace(" ", "_").replace("/", "_") 
            with open(os.path.join(user_dir, f'{sanitized_dish_name}.json'), 'w', encoding='utf-8') as json_file:
                json.dump(evaluations_list, json_file, ensure_ascii=False, indent=4)

    return jsonify({"message": "Data exported successfully for all users!"})


@app.route('/clear-evaluations', methods=['GET'])
def clear_evaluations():
    try:

        num_rows_deleted = db.session.query(Evaluation).delete()
        

        db.session.commit()
        
        return jsonify({"message": f"Cleared {num_rows_deleted} evaluations successfully!"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
