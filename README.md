# pb-api
Repository for simple REST API using [fastAPI](https://fastapi.tiangolo.com/), exposing probability of default (PD) model.
This model is deployed as a web service on [AWS](https://aws.amazon.com/elasticbeanstalk/) Elastic Beanstalk](https://aws.amazon.com/elasticbeanstalk/).

# Model & Evaluation
A `RandomForest` classifier has been trained on the data found in [data](https://github.com/MarcusElwin/pb-api/tree/main/data) folder.
No feature pruning has been done in this stage instead all features have been used where the following transforms have been made:
1. `Ordinal` encoding of categorical features
2. Standard scaling of numerical features
3. `BinaryLabel` encoding of bool features.

See more under [features](https://github.com/MarcusElwin/pb-api/tree/main/features).

The model has been evaluated using `5-fold` cross-validation and the following metrics:
1. AUC
2. prAUC
3. F1-score, precision & recall

# How to run locally:

To start API locally run the cmd found in the `Makefile`:

```sh
make build # build image
make up # run container
make run # run container in interactive mode
make down # stop containter
make destroy # destroy all containers
make train # run training script
make logs # make logs
make help # get help for each cmd
```

# How to use API
API is exposed as a web service at this address: [http://probdefaultapi-env.eba-b4r3fruv.eu-west-1.elasticbeanstalk.com/](http://probdefaultapi-env.eba-b4r3fruv.eu-west-1.elasticbeanstalk.com/).
API Swagger and Documentation can be found at this address: [http://probdefaultapi-env.eba-b4r3fruv.eu-west-1.elasticbeanstalk.com/docs](http://probdefaultapi-env.eba-b4r3fruv.eu-west-1.elasticbeanstalk.com/docs)

This API has the following routes:
1. `/` is a `GET` route for getting model information, see the example output below
```json
{
"version": "df126280-f563-4fbc-a820-42cff04c874f",
"timestamp": "20221114163452"
}
```

2. `/v1/predict` a `POST` route for predicting on one user

3. `/v1/predict/multiple` a `POST` route for predicting one list of users
