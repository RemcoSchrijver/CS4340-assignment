{ pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
    config.cudaSupport = true;
                          }}:

pkgs.mkShell {
    packages = [
        pkgs.python39
        pkgs.python311Packages.pip
    ];
    # Install cuda for the shell
    name = "cuda-env-shell";
    buildInputs = with pkgs; [
        git gitRepo gnupg autoconf curl
        procps gnumake util-linux m4 gperf unzip
        cudatoolkit linuxPackages.nvidia_x11
        libGLU libGL
        xorg.libXi xorg.libXmu freeglut
        xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
        ncurses5 stdenv.cc binutils
    ];
    shellHook = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
    '';  
    # Fixes problem of not finding libstdc++
    # https://discourse.nixos.org/t/how-to-solve-libstdc-not-found-in-shell-nix/25459/7
    LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH";

}