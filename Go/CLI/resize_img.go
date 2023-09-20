package main


import(
	"fmt"
	"os"
	"github.com/nfnt/resize"
	"flag"
	"image"
	"log"
	"io/ioutil"
	"path/filepath"
	"image/jpeg"
    "image/png"
)

var (
	fimg       string
	img_height int
	img_width  int
	aspect     string
	fname      string

)

func main(){

	flag.StringVar(&fimg, "fimg", "", "folder image path")
	flag.IntVar(&img_height, "h", 640, "height image size")
	flag.IntVar(&img_width, "w", 640, "width image size")
	flag.StringVar(&aspect, "aspect", "of", "aspect ratio")
	flag.StringVar(&fname,"fname", "images", "folder name")
	flag.Parse()

	folder_img := dirwalk(fimg)

	for _, img := range folder_img {
		fmt.Println(img)
		img_data, err := os.Open(img)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return
		}

		imgs, _,  err := image.Decode(img_data)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(imgs)

		// 画像形式のチェック
		_, format, err := image.DecodeConfig(img_data)
		if err != nil {
			log.Fatal(err)
		}
		img_data.Seek(0, 0) // ファイルポインタを先頭に戻す

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

		
		x_ratio := float64(img_width) / float64(x_width)
		y_ratio :=  float64(img_height) / float64(y_height)

		img_data.Close()
		output, err := os.Create(img)
		if err != nil {
			fmt.Println(err)
		}

		if aspect == "on" {
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


		}else {
			resizedImg := resize.Resize(uint(img_width), uint(img_height), imgs, resize.NearestNeighbor)
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
