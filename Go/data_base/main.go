package main


import (
	"database/sql"
	"fmt"
	"log"
  
	_ "github.com/mattn/go-sqlite3"
  )
  
func main() {
  	
	db, err := sql.Open("sqlite3", "./test.db")
	if err != nil {
		panic(err)
	}
	
	// Create table
	_, err = db.Exec(
		`CREATE TABLE IF NOT EXISTS "BOOKS" ("ID" INTEGER PRIMARY KEY, "TITLE" VARCHAR(255))`,
	)
	if err != nil {
		panic(err)
	}

	// Insert data
	res, err := db.Exec(
		`INSERT INTO BOOKS (ID, TITLE) VALUES (?, ?)`,
		123,
		"title",
	)
	if err != nil {
		panic(err)
	}
	
	// Get ID
	id, err := res.LastInsertId()
	if err != nil {
		panic(err)
	}
	fmt.Println(id)

	// Get multi record
	rows, err := db.Query(
		`SELECT * FROM BOOKS`,
	)
	if err != nil {
		panic(err)
	}
	// fmt.Println(rows)

	// defer rows.Close()
	for rows.Next() {
		var id int
		var title string
	  
		// Get cursor value
		if err := rows.Scan(&id, &title); err != nil {
		  log.Fatal("rows.Scan()", err)
		  return
		}
	  
		fmt.Printf("id: %d, title: %s\n", id, title)
	  }

  }