package main

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	counter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "myapp_processed_total",
			Help: "Total number of processed requests",
		},
	)
)

func main() {
	// Prometheusのメトリクスを登録
	prometheus.MustRegister(counter)

	// HTTPハンドラの登録
	http.Handle("/metrics", promhttp.Handler())

	// メトリクスの増加
	go func() {
		for {
			counter.Inc()
			fmt.Println("Processed request")
		}
	}()

	// サーバーの起動
	fmt.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		panic(err)
	}
}
