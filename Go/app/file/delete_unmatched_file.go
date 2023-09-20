package file

import (
    "fmt"
    // "log"
    "net/http"
    "path/filepath"
    "os"
)



func Delete_umatched_file(w http.ResponseWriter, r *http.Request){

    File_path := getFilepath()

    if r.Method != "POST" {
		http.Error(w, "処理を終了します。", http.StatusMethodNotAllowed)
		return
	}
    r.ParseForm()
    if _, ok := r.Form["delete_umatched_file"]; ok {
        fmt.Fprintf(w, "チェックボックスがチェックされました！")
    } else {
        fmt.Fprintf(w, "チェックボックスがチェックされていません。")
    }


    var img_list []string
    var labe_list []string
	for _, s := range File_path{
		if filepath.Ext(s) == ".jpg" || filepath.Ext(s) == ".jpeg" || filepath.Ext(s) == "png" {
            img_list = append(img_list,s)

        }else if filepath.Ext(s) == ".txt" || filepath.Ext(s) == ".xml" {
            labe_list = append(labe_list,s)
        }
        
    }

    var img_filename []string
	var image_base_name []string
    var image_extension []string
	for _, s := range img_list{
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
        image_extension = append(image_extension, s[end:])
	}


    var labe_filename []string
	var label_base_name []string
    var label_extension []string
	for _, s := range labe_list{
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
        label_extension = append(label_extension, s[end:])
	}

	duplicates := make(map[string]bool)


    if len(labe_filename) > len(img_filename) {
        for _, item := range img_filename {
            duplicates[item] = true
        }
    
        for _, item := range labe_filename {
            if duplicates[item] {
                fmt.Println(item)
            }else{
                fmt.Println(image_base_name[0])
                os.Remove(image_base_name[0] + item + image_extension[0])
                os.Remove(label_base_name[0] + item + label_extension[0])
            }
        }

    }else {
        for _, item := range labe_filename {
            duplicates[item] = true
        }
    
        for _, item := range img_filename {
            if duplicates[item] {
                fmt.Println(item)
            }else{
                fmt.Println(image_base_name[0])
                os.Remove(image_base_name[0] + item + image_extension[0])
                os.Remove(label_base_name[0] + item + label_extension[0])
            }
        }
    }

}