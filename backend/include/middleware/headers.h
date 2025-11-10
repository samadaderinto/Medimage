#include "crow.h"
#include "crow/middlewares/cors.h"

class HeaderCheckMiddleware
{
public:
    
    struct context
    {};

    
    void before_handle(crow::request& req, crow::response& res, context& ctx)
    {
        
        std::string auth_header = req.get_header_value("Authorization");

        
        if (auth_header.empty() || auth_header.substr(0, 7) != "Bearer ")
        {
            
            res.code = 401;
            res.write("Unauthorized: Missing or invalid Authorization header");
            res.end();
        }
        
    }

    void after_handle(crow::request& req, crow::response& res, context& ctx)
    {
        
        CROW_LOG_INFO << "Request for " << req.url << " handled with status " << res.code;
    }
};
