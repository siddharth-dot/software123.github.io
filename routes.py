from flask import render_template, url_for, flash, redirect, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproject import app, db, bcrypt
from pyproject.forms import RegistrationForm, LoginForm
from pyproject.models import User
from pyproject.models import DiabetesInput
from flask_login import login_user, current_user, logout_user, login_required
features = []
model_id  = ""

@app.route('/')
@app.route('/home')
def home():
    return render_template('home_page.html')

@app.route('/preview')
def preview():
    return render_template('preview.html')


@app.route("/register", methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('diabetesinfo'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('diabetesinfo'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            flash(f'Welcome Back!','success')
            return redirect(next_page) if next_page else redirect(url_for('diabetesinfo'))
            flash('Welcome Back!','success')
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')



@app.route('/diabetesinfo' , methods=['GET','POST'])

def diabetesinfo():
    global features
    global model_id
    if request.method =="POST":
        req=request.form
        name=req["name"]
        Pregnancies=req["pregnancies"]
        features.insert(0,Pregnancies)
        Glucose=req["glucose"]
        features.insert(1,Glucose)
        BP=req["bp"]
        features.insert(2,BP)
        Skin=req["skinthickness"]
        features.insert(3,Skin)
        Insulin=req["insulin"]
        features.insert(4,Insulin)
        BMI=req["bmi"]
        features.insert(5,BMI)
        DPF=req["dpf"]
        features.insert(6,DPF)
        age=req["age"]
        features.insert(7,age)
        model_id=int(req["model"])
        features.insert(8,model_id)
        print(model_id)
        print(features)
        input1 = DiabetesInput(user_id=current_user.id, Name=req["name"], Pregnancies=Pregnancies, BloodPressure=BP, Insulin=Insulin, DPF=DPF, GLUCOSE=Glucose, Skin=Skin, BMI=BMI, AGE=age, Cmodel=model_id)
        db.session.add(input1)
        db.session.commit()
        return redirect(url_for('diabetesprediction'))
    return render_template('diabetesinfo.html')

@app.route("/diabetesprediction")
def diabetesprediction():
    Diabetes=db.session.query(DiabetesInput).order_by(DiabetesInput.uid.desc()).first()
    
    print(features)
    input_list = features[-9:]
    input_list = [int(x) for x in input_list]
    print(input_list)
    model = input_list[-1]
    print(model)
    input_list = input_list[:-1]
    model_id = features[-1]
    print(model_id)
    return render_template('diabetesprediction.html', Diabetes=Diabetes, mylist=mylist(input_list,model,model_id))
    

def mylist(arr,a,model_id):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    arr = [np.array(arr)]
    print(arr)
    
    dataset =pd.read_csv('pyproject/data/diabetes.csv')
    X=dataset.iloc[:,0:8].values
    y=dataset.iloc[:,8].values
    
    from sklearn.preprocessing import LabelEncoder
    y= LabelEncoder().fit_transform(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    sc_X= StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)

    if(model_id==1):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train,y_train)
    elif(model_id==2):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
        classifier.fit(X_train,y_train)
    elif(model_id==3):
        from sklearn.svm import SVC
        classifier = SVC(kernel ='rbf', random_state=0)
        classifier.fit(X_train,y_train)
    elif(model_id==4):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)
    elif(model_id==5):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion ='entropy',random_state=0)
        classifier.fit(X_train,y_train)
    else:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
        classifier.fit(X_train,y_train)
    
    Y_pred = classifier.predict(X_test)
    print(X_test)
    
    from  sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,Y_pred)
    
    pred = classifier.predict(arr)[0]
    print(type(pred))
    print(pred)
    if pred == 1:
        ans = "The person will get Diabetes"
    else:
        ans = "The person will not get Diabetes"    

    a = round((cm[0][0]+cm[1][1])*100/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]),2)
    b = round((cm[0][0])*100/(cm[0][0]+cm[1][0]),2)
    v = round((cm[0][0])*100/(cm[0][0]+cm[0][1]),2)
    d = round(2*(v*b)/(v+b),2)
    result = [a,b,v,d]
    print(result)
    result.append(ans)

    from sklearn.decomposition import PCA
    pca= PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    explained_variance= pca.explained_variance_ratio_

    print(explained_variance)

    if(model_id==0):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(X_train,y_train)
    elif(model_id==1):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)
        classifier.fit(X_train,y_train)
    elif(model_id==2):
        from sklearn.svm import SVC
        classifier = SVC(kernel ='rbf', random_state=0)
        classifier.fit(X_train,y_train)
    elif(model_id==3):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train,y_train)
    elif(model_id==4):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion ='entropy',random_state=0)
        classifier.fit(X_train,y_train)
    else:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)
        classifier.fit(X_train,y_train)

    Y_pred = classifier.predict(X_test)

    from matplotlib.colors import ListedColormap
    X_set,y_set=X_test,y_test
    X1,X2= np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
    plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap =ListedColormap(('red','green')))
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
    plt.title('PCA for Selected Model')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.legend()

    return result   

@app.route('/diabetesprediction/response', methods=['GET','POST'])
def diabetes_response():
    res = [x for x in request.form.values()]
    print(res)

    return render_template('home_page.html')