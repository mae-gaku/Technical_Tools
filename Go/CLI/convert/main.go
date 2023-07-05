package main

import (
	"main/conversion"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"io/ioutil"
)

var (
	extension string
	imagepath string
	dirpath   string
	fname     string
)

func main() {

	flag.StringVar(&extension, "ext", "jpeg", "file extension")
	flag.StringVar(&imagepath, "fimg", "", "image file path")
	flag.StringVar(&fname,"fname" ,"image", "folder name")
	flag.Parse()
	err := conversion.ExtensionCheck(extension)
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}
	img_ext := extension
	image_folder := dirwalk(imagepath)
	image_folder_name := fname

	// current directory
	base_file_path, err := os.Getwd()
	if err != nil {
		panic(err)
	}

	// Create folder
	folder_path := base_file_path + "/" + image_folder_name
	if err := os.Mkdir(folder_path, 0777); err != nil {
		fmt.Println(err)
	}

	for _, img := range image_folder{

		err = conversion.FilepathCheck(img)
		if err != nil {
			fmt.Println("Error:", err)
			os.Exit(1)
		}
		f := filepath.Ext(img)
		err = conversion.FileExtCheck(f)
		if err != nil {
			fmt.Println("Error:", err)
			os.Exit(1)
		}
	
		fmt.Println("Converting...")
	
		begin, end := 0, 0
		for i := len(img) - 1; i >= 0; i-- {
			if img[i] == '/' {
				begin = i + 1
				break
			}
			if img[i] == '.' {
				end = i
			}
		}
		dirpath = folder_path + "/"+img[begin:end] + "." + img_ext

		err = conversion.FileExtension(extension, img, dirpath)
		if err != nil {
			fmt.Println("Error:", err)
			os.Exit(1)
		}

	}

}


func dirwalk(dir string) []string {
    files, err := ioutil.ReadDir(dir)
    if err != nil {
        panic(err)
    }

    var paths []string
    for _, file := range files {
        if file.IsDir() {
            paths = append(paths, dirwalk(filepath.Join(dir, file.Name()))...)
            continue
        }
        paths = append(paths, filepath.Join(dir, file.Name()))
    }

    return paths
}
