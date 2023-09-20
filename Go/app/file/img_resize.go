package file

import (
	"fmt"
	"net/http"
    "os"
	"image"
	"log"
	"image/jpeg"
    "image/png"
	"path/filepath"
	"github.com/nfnt/resize"
	"strconv"
)



func Img_resize(w http.ResponseWriter, r *http.Request){
    File_path := getFilepath()
    if r.Method != "POST" {
		http.Error(w, "処理を終了します。", http.StatusMethodNotAllowed)
		return
	}
    r.ParseForm()

	check := r.FormValue("image")
	img_width := r.FormValue("width")
	img_height := r.FormValue("height")

	if check == "yes" && img_width != "" && img_height != "" {
		fmt.Fprintf(w, "画像がリサイズされました")
	} else {
		http.Error(w, "処理を終了します。", http.StatusMethodNotAllowed)
		return
	}

	for _, img := range File_path {
		if filepath.Ext(img) == ".jpg" || filepath.Ext(img) == ".jpeg" || filepath.Ext(img) == "png" {
			img_data, err := os.Open(img)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				return
			}
	
			_, format, err := image.DecodeConfig(img_data)
			if err != nil {
				log.Fatal(err)
			}
			img_data.Seek(0, 0) 
	
			var imgs image.Image
			switch format {
			case "jpg","jpeg":
				imgs, err = jpeg.Decode(img_data)
			case "png":
				imgs, err = png.Decode(img_data)
			default:
				log.Fatalf("unsupported format: %s", format)
			}
			if err != nil {
				log.Fatal(err)
			}
	
			x_width := imgs.Bounds().Dx()
			y_height := imgs.Bounds().Dy()
			
			fmt.Println(x_width)
			fmt.Println(y_height)

			img_width, err :=  strconv.ParseFloat(img_width, 64)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
                return
			}

			img_height, err :=  strconv.ParseFloat(img_height, 64)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
                return
			}

			x_ratio := float64(img_width) / float64(x_width)
			y_ratio :=  float64(img_height) / float64(y_height)
	
			img_data.Close()
			output, err := os.Create(img)
			if err != nil {
				fmt.Println(err)
			}
	

			if x_ratio < y_ratio {
				resizedImg := resize.Resize(uint(img_width), uint(int(float64(y_height) * x_ratio)), imgs, resize.NearestNeighbor)
				switch format {
				case "png":
					if err := png.Encode(output, resizedImg); err != nil {
						log.Fatal(err)
					}
				case "jpeg", "jpg":
					opts := &jpeg.Options{Quality: 100}
					if err := jpeg.Encode(output, resizedImg, opts); err != nil {
						log.Fatal(err)
					}
				default:
					if err := png.Encode(output, resizedImg); err != nil {
						log.Fatal(err)
					}
				}

			}else{
				resizedImg := resize.Resize(uint(int(float64(x_width) * y_ratio)), uint(img_height), imgs, resize.NearestNeighbor)
				switch format {
				case "png":
					if err := png.Encode(output, resizedImg); err != nil {
						log.Fatal(err)
					}
				case "jpeg", "jpg":
					opts := &jpeg.Options{Quality: 100}
					if err := jpeg.Encode(output, resizedImg, opts); err != nil {
						log.Fatal(err)
					}
				default:
					if err := png.Encode(output, resizedImg); err != nil {
						log.Fatal(err)
					}
				}

			}


            }
            
	}

}