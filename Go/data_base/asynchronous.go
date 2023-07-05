package main

import (
    "fmt"
    "time"
)

var g int

func change(num int) {
    time.Sleep(100 * time.Millisecond)
    g = num
}

func main() {
    change(1)
    go change(2)
    fmt.Println(g)  // 1 (2が実行される前にだいたい到達する)

    time.Sleep(100 * time.Millisecond)
    fmt.Println(g)  // 1 or 2 (スレッドの状況によって実行順序が変わる)
}