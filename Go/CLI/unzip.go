package main

import (
    "archive/zip"
    "fmt"
    "io"
    "os"
	"path/filepath"
)

func main() {
    zipFilePath := "/media/sf_virtualbox/images.zip"
    destDirectory := "/home/gaku/Go/sample_cli/zip"

    // zipファイルを開く
    r, err := zip.OpenReader(zipFilePath)
    if err != nil {
        panic(err)
    }
    defer r.Close()

    // ファイルを展開する
    for _, f := range r.File {
        // ファイルを開く
        rc, err := f.Open()
        if err != nil {
            panic(err)
        }
        defer rc.Close()

        // 出力先のファイルを作成
        path := filepath.Join(destDirectory, f.Name)
        if f.FileInfo().IsDir() {
            os.MkdirAll(path, f.Mode())
        } else {
            os.MkdirAll(filepath.Dir(path), f.Mode())
            outFile, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
            if err != nil {
                panic(err)
            }
            defer outFile.Close()

            // ファイルの内容をコピー
            _, err = io.Copy(outFile, rc)
            if err != nil {
                panic(err)
            }
        }
    }

    fmt.Println("Extraction complete.")
}
