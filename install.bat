py -m venv venv
call .\venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/kashif/diffusers.git@a3dc21385b7386beb3dab3a9845962ede6765887 --force
pause
