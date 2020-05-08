# Zillow data clustering project
## Allen Jiang and Cameron Taylor

To reproduce our project, you must run:
    df = acquire.zillow_data()
    df = acquire.acquire_data()
    df = prepare_data(),
then split your data and go from there. 
You must also have an env file with the function: 
    def url(db_name):
        from env import host, user, password
        url = f'mysql+pymysql://{user}:{password}@{host}/{db_name}'
        return url
This function allows you to access the database with the Zillow data on it