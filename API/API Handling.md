APIs are handled using pythons requests (with an S) module
1. pip install requests
2. read documentation for endpoint
    ## What Are API Endpoints?
		1. A URL defines the data you interact with on a web server. 
		2. Much like a web page URL is connected to a single page, an endpoint URL 
		   is linked to particular resources within an API. 
		3. Therefore, an endpoint may be described as a digital location where an
		    API receives inquiries about a particular resource on its server\
3. to fetch data from website from a website as json inside a dictionary, use -
   ## GET requests
   ```
		import requests
		url='https://api.example.com/data'
		response = requests.get(url=url)
		data = response.json()
		print(data)
	```
	response.json() method stores the response data in a dictionary object; note that this only 
	works because the result is written in JSON format – an error would have been raised otherwise.
	## Headers in GET requests
	- Headers convey metadata about the request or provide authentication details.
	- typically in the form of key-value pairs. 
	#### Common Use Cases:
	- **Content Negotiation:** Specify acceptable response formats, such as JSON or XML, using `Accept` headers.
		 #### Example
		```

		import requests
		headers = {"accept": "application/json"}
		url='https://api.example.com/data'
		
		response = requests.get(url=url, headers=headers )
		data = response.json()
		print(data)
		```
	- This tells the server that the client expects the response in JSON format.
	### **`params` Parameter**
	- These parameters are appended to the URL after the `?` symbol 
	- Ued to filter, search, or paginate data.
	#### Common Use Cases:
	- **Filtering:** Specify criteria for the data to be returned (e.g., `query="human"`).
	- **Pagination:** Control the number of results and pages (e.g., `page="1"` and `limit="10"`).
	- **Search Queries:** Define search terms or keywords.
	  `querystring = {"page":"1", "limit":"10", "query":"human"}`
	  - The query string is appended to the URL as:  
    `https://api.freeapi.app/api/v1/public/quotes?page=1&limit=10&query=human`
	- This instructs the server to return the first page with a limit of 10 results, filtered by the keyword "human."

4.  submit data to an api
    ## POST request 
	   post_response = requests.post(url_post, json=new_data)
5.  basic authentication
	```
	  from reqquests.auth import HTTPBasicAuth
	  private_url_response = requests.get(
	  url=private_url,auth=HTTPBasicAuth(github_username, token) )
	```
    [`HTTPBasicAuth`](https://requests.readthedocs.io/en/latest/_modules/requests/auth/#:~:text=class%20HTTPBasicAuth(AuthBase)%3A) object from `requests.auth`. This object attaches HTTP basic authentication to the given request object—it’s essentially the same as typing your username and password into a website.
6. 



