package main

import (
    "fmt"
    "io/ioutil"
    "path/filepath"
	"os"
	"flag"
)



func main() {
	var (
		image_file_path = flag.String("image", "/path/to/*", "image file path")
        label_file_path = flag.String("label", "/path/to/*", "label file path")
		image_extension = flag.String("imgext", ".jpg", "file extension")
		label_extension = flag.String("labext", ".txt", "file extension")
    )

	flag.Parse()

	image_file_path_1 := *image_file_path
	label_file_path_1 := *label_file_path
	img_extension := *image_extension
	lab_extension := *label_extension
	
	image_path := dirwalk(image_file_path_1)
	label_path := dirwalk(label_file_path_1)

	var img_filename []string
	var image_base_name []string
	for _, s := range image_path{
		begin, end := 0, 0
		for i := len(s) - 1; i >= 0; i-- {
			if s[i] == '/' {
				begin = i + 1
				break
			}
			if s[i] == '.' {
				end = i
			}
		}
		img_filename = append(img_filename, s[begin:end])
		image_base_name = append(image_base_name, s[:begin])
	}
	
	var labe_filename []string
	var label_base_name []string
	for _, s := range label_path{
		begin, end := 0, 0
		for i := len(s) - 1; i >= 0; i-- {
			if s[i] == '/' {
				begin = i + 1
				break
			}
			if s[i] == '.' {
				end = i
			}
		}
		labe_filename = append(labe_filename, s[begin:end])
		label_base_name = append(label_base_name, s[:begin])
	}

	duplicates := make(map[string]bool)

	for _, item := range img_filename {
        duplicates[item] = true
    }

	for _, item := range labe_filename {
        if duplicates[item] {
            fmt.Println(item)
        }else{
			os.Remove(image_base_name[0] + item + img_extension)
			os.Remove(label_base_name[0] + item + lab_extension)
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