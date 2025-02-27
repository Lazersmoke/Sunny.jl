# Trick to display a string as HTML in various environments
struct SunnyViewer
    html_str::String
end
# See Makie's logic for auto-generating these:
# https://github.com/JuliaPlots/Makie.jl/blob/master/WGLMakie/src/display.jl#L12
Base.show(io::IO, ::MIME"juliavscode/html", sv::SunnyViewer) = print(io, sv.html_str)
Base.show(io::IO, ::MIME"text/html", sv::SunnyViewer) = print(io, sv.html_str)
Base.show(io::IO, ::MIME"text/plain", sv::SunnyViewer) = print(io, "SunnyViewer(...)")

function wrap_html(html::String)
    return open(joinpath(@__DIR__, "assets/standalone_wrapper.html"), "r") do io
        wrapper = read(io, String)
        replace(wrapper, "\$PAYLOAD" => html)
    end
end

"""
    browser(html_str; dir)

Launch a system browser to display the provided HTML string or SunnyViewer. If a
directory `dir` is provided, an HTML file will be written at that location.
"""
function browser(html_str::String; dir=Nothing)
    if Sys.islinux() && dir==Nothing
        println("""If your browser cannot access `/tmp`, consider specifying a directory:
                       browser(html_str; dir="~")""")
    end

    is_tempdir = (dir == Nothing)
    dir = is_tempdir ?  mktempdir() : expanduser(dir)
    path = joinpath(dir, "SunnyGfx.html")
    !is_tempdir && println("Writing `$path`.")
    open(path, "w") do io
        write(io, html_str)
    end
    try
        if Sys.isapple()
            run(`open file://$path`)
        elseif Sys.islinux() || Sys.isbsd()
            run(`xdg-open file://$path`)
        elseif Sys.iswindows()
            # Note the three backslashes `file:///`. It appears unnecessary to swap backslash/slash: \ => /
            # Source: https://stackoverflow.com/a/18246357/500314
            # path = replace(path, "\\" => "/")
            run(`start file:///$path`)
        else
            error("Unsupported system.")
        end
    catch e
        error("Failed to open the generated HTML file $path\n",
              "Error: ", sprint(Base.showerror, e))
    end
end

function browser(sv::SunnyViewer; dir=Nothing)
    browser(wrap_html(sv.html_str); dir)
end

function offline_viewers()
    three_src = inflate_gzip(joinpath(@__DIR__, "assets/three.js.min-143-dev.gz"))
    orbit_controls_src = inflate_gzip(joinpath(@__DIR__, "assets/OrbitControls.js-r142.gz"))
    html_str = """
        3D graphics package for Jupyter notebooks has been installed.
        <script>
        (function() {
            // Hack to make "three.js" write to `exports`, regardless of the module system.
            let exports = {}, module = {};
            $three_src
            // Hack to make "OrbitControls.js" write to `exports.OrbitControls`.
            let THREE = exports;
            $orbit_controls_src
            // Save library in globalThis.SUNNY_THREE.
            globalThis.SUNNY_THREE = exports;
        })();
        </script>
    """
    SunnyViewer(html_str)
end


# For debugging inside VSCode:
#   Developer: Toggle Developer Tools
# For possible webview features:
#   https://code.visualstudio.com/api/extension-guides/webview