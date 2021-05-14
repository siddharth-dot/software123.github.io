from pyproject import db, login_manager
from flask_login import UserMixin

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    diab = db.relationship('DiabetesInput', lazy=True)

def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

class DiabetesInput(db.Model):
    uid = db.Column(db.Integer, primary_key=True)
    user_id=db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    Name =db.Column(db.String(20), nullable=False)
    Pregnancies= db.Column(db.Integer, nullable=False)
    BloodPressure = db.Column(db.Integer, nullable=False)
    Insulin= db.Column(db.Integer, nullable=False)
    DPF= db.Column(db.Integer, nullable=False)
    GLUCOSE = db.Column(db.Integer, nullable=False)
    Skin = db.Column(db.Integer, nullable=False)
    BMI = db.Column(db.Integer, nullable=False)
    AGE = db.Column(db.Integer, nullable=False)
    Cmodel = db.Column(db.String(60), nullable=False)

def __repr__(self):
        return f"DiabetesInput('{self.uid}', '{self.Name}', '{self.BloodPressure}', '{self.Insulin}', '{self.DPF}', '{self.GLUCOSE}, '{self.Skin}', '{self.BMI}', '{self.AGE}', '{self.Cmodel}')"

    