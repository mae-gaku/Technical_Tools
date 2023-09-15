// upload.go
package file

import (
	// "fmt"
	"io"
	"net/http"
	"os"
	"archive/zip"
)

var File_path []string

func UploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "処理を終了します。", http.StatusMethodNotAllowed)
		return 
	}

	file, fileHeader, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	defer file.Close()

	// Check zip file
	if fileHeader.Header.Get("Content-Type") != "application/zip" {
		http.Error(w, "Invalid file format", http.StatusBadRequest)
		return
	}

	// Unzip
	reader, err := zip.NewReader(file, fileHeader.Size)
	
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for _, f := range reader.File {
		err = extractFile(f,w)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return 
		}
	}

	http.Redirect(w, r, "/process", http.StatusSeeOther)
}




func extractFile(f *zip.File, w http.ResponseWriter) error {
    // ファイルを解凍する
    rc, err := f.Open()
    if err != nil {
        return err
    }
    defer rc.Close()

    // 解凍先のファイルを作成する
    path := f.Name
	File_path = append(File_path, path)
	// fmt.Println(path)
    if f.FileInfo().IsDir() {
        os.MkdirAll(path, 0777)
    } else {
        file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0777)
        if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			// return
        }
        defer file.Close()

        // copy file
        _, err = io.Copy(file, rc)
        if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			// return
        }
    }

    return nil
}


func getFilepath() []string {
    File_path := File_path
	
    return File_path
}
