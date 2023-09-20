import pandas as pd

POSES = {
        "Stand": 0,
        "fall down":1,
        "Sit":2
        }

class Pose():
    def pose_db(self,kpt_coord_dict):
        
        key_points = [
            'class','nose', 'Right eye', 'Left eye', 'Right ear','Left ear', 
            'Right shoulder', 'Left shoulder','Right elbow','Left elbow',
            'Right hand','Left hand','Right hip','Left hip','Right knee','Left knee',
            'Right leg','Left leg']

        header = {}
        for key_point in key_points:
            if key_point == 'class':
                header['{}'.format(key_point)] = []
            else:
                header['{} x'.format(key_point)] = []
                header['{} y'.format(key_point)] = []


        # for pose in POSES:
            df = pd.DataFrame(header)
            # for image_path in name_num:
        try:
            results = kpt_coord_dict
            
            if len(results) > 0:
                results = kpt_coord_dict
                # Filter only desired key points
                results = {key: value for key, value in results.items() if key in key_points}

                new_row = pd.DataFrame(header)
                for key, value in results.items():
                    if key not in key_points:
                        continue
                    new_row['{} x'.format(key)] = [value[0]]
                    new_row['{} y'.format(key)] = value[1]

                df = df.append(new_row, ignore_index=True)
                df.fillna({'class': 0},inplace=True)

            else:
                print("b")
        except Exception as e:
                print("a")

            # df.to_csv('{}.csv'.format(pose))

        return df

    
    def db_svm(self,pose_kpt_db,model):

        from sklearn.preprocessing import StandardScaler
 
        data = pose_kpt_db

        X = data[['nose x','nose y','Right eye x','Right eye y','Left eye x',
        'Left eye y','Right ear x','Right ear y','Left ear x','Left ear y',
        'Right shoulder x','Right shoulder y','Left shoulder x','Left shoulder y', 
        'Right elbow x','Right elbow y','Left elbow x','Left elbow y','Right hand x',
        'Right hand y','Left hand x','Left hand y','Right hip x','Right hip y',
        'Left hip x','Left hip y','Right knee x','Right knee y','Left knee x','Left knee y',
        'Right leg x','Right leg y','Left leg x','Left leg y']]

        y = data['class']

        # sc = StandardScaler()
        # sc.fit(X)
        # X_test_std = sc.transform(X)

        predicted = model.predict(X) 

        return predicted




