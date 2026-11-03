/* Linux V4L2 control-only helper. Does not request buffers, capture, or actuate.
 * Default: read controls. --freeze-exposure: use the reported current absolute
 * exposure with manual mode. Built against the board's own kernel headers.
 */
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static int get_control(int fd, unsigned int id, int *value) {
    struct v4l2_control c = {0};
    c.id = id;
    if (ioctl(fd, VIDIOC_G_CTRL, &c) < 0) return -1;
    *value = c.value;
    return 0;
}

static int set_control(int fd, unsigned int id, int value) {
    struct v4l2_control c = {0};
    c.id = id;
    c.value = value;
    return ioctl(fd, VIDIOC_S_CTRL, &c);
}

static void print_control(int fd, const char *name, unsigned int id) {
    int value;
    printf("\"%s\":", name);
    if (get_control(fd, id, &value) == 0) printf("%d", value);
    else printf("null");
}

int main(int argc, char **argv) {
    const char *device = "/dev/video0";
    int freeze = 0, restore_auto = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--freeze-exposure") == 0) freeze = 1;
        else if (strcmp(argv[i], "--restore-auto-exposure") == 0) restore_auto = 1;
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
        else { fprintf(stderr, "Usage: camera_controls [--device /dev/video0] [--freeze-exposure]\n"); return 2; }
    }
    int fd = open(device, O_RDWR | O_NONBLOCK);
    if (fd < 0) { perror("open camera control device"); return 1; }
    int old_mode = -1, absolute = -1;
    get_control(fd, V4L2_CID_EXPOSURE_AUTO, &old_mode);
    get_control(fd, V4L2_CID_EXPOSURE_ABSOLUTE, &absolute);
    if (restore_auto) {
        if (freeze || set_control(fd, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_APERTURE_PRIORITY) < 0) {
            fprintf(stderr, "Could not restore original aperture-priority auto mode\n"); close(fd); return 1;
        }
    }
    if (freeze) {
        if (old_mode != V4L2_EXPOSURE_MANUAL) {
            fprintf(stderr, "Automatic exposure readback may be a cached manual value; refusing to freeze it\n");
            close(fd); return 1;
        }
        if (old_mode < 0 || absolute <= 0) {
            fprintf(stderr, "Camera does not report an absolute exposure; no settings changed\n");
            close(fd); return 1;
        }
        if (set_control(fd, V4L2_CID_EXPOSURE_AUTO, V4L2_EXPOSURE_MANUAL) < 0) {
            perror("set manual exposure"); close(fd); return 1;
        }
        if (set_control(fd, V4L2_CID_EXPOSURE_ABSOLUTE, absolute) < 0) {
            int saved_errno = errno;
            set_control(fd, V4L2_CID_EXPOSURE_AUTO, old_mode);
            errno = saved_errno; perror("preserve absolute exposure"); close(fd); return 1;
        }
        int now = -1;
        if (get_control(fd, V4L2_CID_EXPOSURE_AUTO, &now) < 0 || now != V4L2_EXPOSURE_MANUAL) {
            fprintf(stderr, "Manual exposure readback not confirmed\n"); close(fd); return 1;
        }
    }
    printf("{\"control_only\":true,\"freeze_requested\":%s,\"previous_exposure_mode\":%d,", freeze ? "true" : "false", old_mode);
    print_control(fd, "exposure_mode", V4L2_CID_EXPOSURE_AUTO); printf(",");
    print_control(fd, "exposure_absolute", V4L2_CID_EXPOSURE_ABSOLUTE); printf(",");
    print_control(fd, "auto_white_balance", V4L2_CID_AUTO_WHITE_BALANCE); printf(",");
    print_control(fd, "white_balance_temperature", V4L2_CID_WHITE_BALANCE_TEMPERATURE); printf(",");
    print_control(fd, "brightness", V4L2_CID_BRIGHTNESS); printf(",");
    print_control(fd, "gain", V4L2_CID_GAIN); printf("}\n");
    close(fd);
    return 0;
}
