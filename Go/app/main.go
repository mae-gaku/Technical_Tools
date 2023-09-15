// main.go
package main

import (
    "fmt"
    "log"
    "net/http"
	"xml/file"
    // "xml/check"
)

func indexHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Add("Content-Type", "text/html")
    http.ServeFile(w, r, "index.html")
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	file.UploadHandler(w, r)
    
}

func processHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Add("Content-Type", "text/html")
    http.ServeFile(w, r, "process.html")

}

func xml_checkbox(w http.ResponseWriter, r *http.Request) {
    file.Xml_checkbox(w,r)
}

func img_resize(w http.ResponseWriter, r *http.Request) {
    file.Img_resize(w,r)
}


func delete_umatched_file(w http.ResponseWriter, r *http.Request) {
    file.Delete_umatched_file(w,r)
}


func setupRoutes() {
    mux := http.NewServeMux()
    mux.HandleFunc("/", indexHandler)
    mux.HandleFunc("/upload", uploadHandler)
    mux.HandleFunc("/process", processHandler)
    mux.HandleFunc("/xml_checkbox",xml_checkbox)
    mux.HandleFunc("/img_resize",img_resize)
    mux.HandleFunc("/delete_umatched_file",delete_umatched_file)

    if err := http.ListenAndServe(":8080", mux); err != nil {
        log.Fatal(err)
    }
}


func main() {
    fmt.Println("ファイルアップロード開始")
    setupRoutes()
}