package main

import (
    "fmt"
    "io/ioutil"
    "path/filepath"
	"os"
	"encoding/xml"
	"flag"
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


type stringFlags []string

func (v *stringFlags) String() string {
	return fmt.Sprintf("%v", options)
}

func (s *stringFlags) Set(v string) error {
	*s = append(*s, v)
	return nil
}

var options stringFlags

func main() {
	var (
        label_file_path = flag.String("xmlpath", "/path/to/*", "xml file path")
		folder_path = flag.String("fname", "label1", "folder name")
    )
	flag.Var(&options, "cls", "class name")

	flag.Parse()

	cls_swi := options

	// xml label
	label_file_path_1 := *label_file_path
	folder_name_path := *folder_path
    fmt.Println(dirwalk(label_file_path_1))
	xml_path := dirwalk(label_file_path_1)

	for _, s := range xml_path{
		// fmt.Println(i,s)
		// s = `"` + s + `"`

		// xml
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
		// fmt.Println(v)

		// current directory
		base_file_path, err := os.Getwd()
		if err != nil {
			panic(err)
		}

		// Create folder
		folder_path := base_file_path + "/" + folder_name_path
		if err := os.Mkdir(folder_path, 0777); err != nil {
			fmt.Println(err)
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

			// if tmp == ob_name{
			// 	// txt file write
			// 	_, err = fmt.Fprintln(txt_file,"0", el_1, el_2, el_3, el_4)
			// 	if err != nil {
			// 		fmt.Println(err)
			// }

			// Create classes.txt

			cls_txt_file, err := os.Create(folder_path + "/" + "classes.txt")
			if err != nil {
				fmt.Println(err)
			}
			defer txt_file.Close()

			for _, s := range cls_swi{
				_, err = fmt.Fprintln(cls_txt_file,s)
				if err != nil {
					fmt.Println(err)
				}
			}

			for index, value := range cls_swi {
				fmt.Println(index, value)
				if cls_swi[index] == ob_name{
					_, err = fmt.Fprintln(txt_file,index, el_1, el_2, el_3, el_4)
					if err != nil {
						fmt.Println(err)
				}else{
					fmt.Println("Error")
				}
				}
			}

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
