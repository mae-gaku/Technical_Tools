FROM golang:1.14

RUN mkdir /app

WORKDIR /app

COPY ./ ./

RUN chmod -R 777 ./


RUN go build -o app main.go

USER user

CMD ["go", "run", "main.go"]