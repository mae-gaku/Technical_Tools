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
)


var (
	fimg       string
	img_height int
	img_width  int
	fname      string
)


func main() {

	flag.StringVar(&fimg, "fimg", "", "folder image path")
	flag.IntVar(&img_height, "h", 640, "height image size")
	flag.IntVar(&img_width, "w", 640, "width image size")
	flag.StringVar(&fname,"fname", "images", "folder name")
	flag.Parse()

    folder_img := dirwalk(fimg)

    newWidth := img_width
    newHeight := img_height

    for _, imgs := range folder_img {
		fmt.Println(imgs)
		img_data, err := os.Open(imgs)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			return
		}

        img, _,  err := image.Decode(img_data)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(img)
        
        // resize img
        resizedImg := resize.Resize(uint(newWidth), uint(newHeight), img, resize.Lanczos3)

        // バウンディングボックスもリサイズする
        resizedBoxes := resizeBoxes(boxes, origWidth, origHeight, newWidth, newHeight)

        // リサイズ後の画像とバウンディングボックスを保存する
        save(resizedImg, resizedBoxes, filename)

}

}


func resizeBoxes(boxes []box, origWidth, origHeight, newWidth, newHeight float64) []box {
    var resizedBoxes []box
    for _, b := range boxes {
        // バウンディングボックスの座標をリサイズ後の画像の座標系に変換
        newX := b.X * newWidth / origWidth
        newY := b.Y * newHeight / origHeight

        // バウンディングボックスの幅と高さをリサイズ
        newWidth := b.Width * newWidth / origWidth
        newHeight := b.Height * newHeight / origHeight

        // 座標と幅と高さを整数に丸める
        newX = math.Round(newX)
        newY = math.Round(newY)
        newWidth = math.Round(newWidth)
        newHeight = math.Round(newHeight)

        // バウンディングボックスが画像の境界を超えないように調整する
        if newX < 0 {
            newX = 0
        }
        if newY < 0 {
            newY = 0
        }
        if newX+newWidth > newWidth {
            newWidth = newWidth - newX
        }
        if newY+newHeight > newHeight {
            newHeight = newHeight - newY
        }

        // リサイズ後のバウンディングボックスを追加する
        resizedBoxes = append(resizedBoxes, box{
            Class:  b.Class,
            X:      newX,
            Y:      newY,
            Width:  newWidth,
            Height: newHeight,
        })
    }
    return resizedBoxes
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