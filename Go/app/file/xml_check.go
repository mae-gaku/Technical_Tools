package file

import (
    "fmt"
    "net/http"
    "os"
    "path/filepath"
    "encoding/xml"
    "io/ioutil"
    "bufio"
    "strings"
)

type annotation struct {
    Size Size `xml:"size"`
	Object Object `xml:"object"`
}

type Size struct {
    Width int `xml:"width"`
	Height int `xml:"height"`
}

type Object struct {
	Name []string `xml:"name"`
    // Name string `xml:"name"`
	Bndbox []Bndbox `xml:"bndbox"`
	// Bndbox Bndbox `xml:"bndbox"`
}

type Bndbox struct {
    Xmin int `xml:"xmin"`
	Ymin int `xml:"ymin"`
	Xmax int `xml:"xmax"`
	Ymax int `xml:"ymax"`
}



func Xml_checkbox(w http.ResponseWriter, r *http.Request){
    File_path := getFilepath()

    if r.Method != "POST" {
		http.Error(w, "処理を終了します。", http.StatusMethodNotAllowed)
		return
	}
    r.ParseForm()
    if _, ok := r.Form["xml_checkbox"]; ok {
        fmt.Fprintf(w, "チェックボックスがチェックされました！")
    } else {
        fmt.Fprintf(w, "チェックボックスがチェックされていません。")
    }


    // スライスを初期化
    cls_swi := []string{}
    for _, file := range File_path {
        if file[len(file)-4:] == ".txt" {
            fmt.Println(file)
            f, err := os.Open(file)
            if err != nil {
                http.Error(w, err.Error(), http.StatusInternalServerError)
                return
            }
            defer f.Close()

            scanner := bufio.NewScanner(f)
            for scanner.Scan() {
                // 行を分割し、スライスに追加
                cls_swi = append(cls_swi, strings.TrimSpace(scanner.Text()))
            }

            if err := scanner.Err(); err != nil {
                panic(err)
            }
        }
    }

    
    // Create folder
    folder_path := "./" + "TXT"
    if err := os.Mkdir(folder_path, 0777); err != nil {
        fmt.Println(err)
    }

    for _, s := range File_path {
        fmt.Println(s)
        if filepath.Ext(s) == ".xml" {
            xml_file, err := os.Open(s)
            if err != nil {
                panic(err)
            }
            defer xml_file.Close()
            data, err := ioutil.ReadAll(xml_file)
            if err != nil {
                fmt.Printf("error: %v", err)
            }

            v := annotation{}
		
            // var ev annotation
            err = xml.Unmarshal(data, &v)
            if err != nil {
                fmt.Printf("error: %v", err)
                return
            }


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
            
            txt_file_name := folder_path + "/" + s[begin:end] + "." + "txt"
    
            txt_file, err := os.Create(txt_file_name)
            if err != nil {
                fmt.Println(err)
            }
            defer txt_file.Close()

            len_ob := len(v.Object.Name)

            for i := 0; i < len_ob; i++{
                ob_name := v.Object.Name[i]
                ob_width := v.Size.Width
                ob_height := v.Size.Height
                // ob_bndbox := v.Object.Bndbox[i]
                ob_xmin := v.Object.Bndbox[i].Xmin
                ob_ymin := v.Object.Bndbox[i].Ymin
                ob_xmax := v.Object.Bndbox[i].Xmax
                ob_ymax := v.Object.Bndbox[i].Ymax
                
                el_1 := (float64(ob_xmin) + float64(ob_xmax)) / 2 / float64(ob_width)
                el_2 := (float64(ob_ymin) + float64(ob_ymax)) / 2 / float64(ob_height)
                el_3 := (float64(ob_xmax) - float64(ob_xmin)) / float64(ob_width)
                el_4 := (float64(ob_ymax) - float64(ob_ymin)) / float64(ob_height)


                cls_txt_file, err := os.Create(folder_path + "/" + "classes.txt")
                if err != nil {
                    fmt.Println(err)
                }

                for _, s := range cls_swi{
                    _, err = fmt.Fprintln(cls_txt_file,s)
                    if err != nil {
                        fmt.Println(err)
                    }
                }

                for index, _ := range cls_swi {
                    if string(cls_swi[index]) == ob_name {
                        _, err = fmt.Fprintln(txt_file,index, el_1, el_2, el_3, el_4)
                        if err != nil {
                            fmt.Println(err)
                    }else{
                        fmt.Println("Error")
                    }
                    }
                }
            }
        }else {
            fmt.Println("error")
        }
    
   }
}